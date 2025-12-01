//! Random number generator support

use super::{Uint, Word};
use crate::{Limb, NonZero, Random, RandomBits, RandomBitsError, RandomMod, Zero};
use rand_core::{RngCore, TryRngCore};
use subtle::ConstantTimeLess;

impl<const LIMBS: usize> Random for Uint<LIMBS> {
    fn try_random<R: TryRngCore + ?Sized>(rng: &mut R) -> Result<Self, R::Error> {
        let mut limbs = [Limb::ZERO; LIMBS];

        for limb in &mut limbs {
            *limb = Limb::try_random(rng)?
        }

        Ok(limbs.into())
    }
}

/// Fill the given limbs slice with random bits.
///
/// NOTE: Assumes that the limbs in the given slice are zeroed!
///
/// When combined with a platform-independent "4-byte sequential" `rng`, this function is
/// platform-independent. We consider an RNG "`X`-byte sequential" whenever
/// `rng.fill_bytes(&mut bytes[..i]); rng.fill_bytes(&mut bytes[i..])` constructs the same `bytes`,
/// as long as `i` is a multiple of `X`.
/// Note that the `TryRngCore` trait does _not_ require this behaviour from `rng`.
pub(crate) fn random_bits_core<R: TryRngCore + ?Sized>(
    rng: &mut R,
    zeroed_limbs: &mut [Limb],
    bit_length: u32,
) -> Result<(), RandomBitsError<R::Error>> {
    if bit_length == 0 {
        return Ok(());
    }

    let buffer: Word = 0;
    let mut buffer = buffer.to_be_bytes();

    let nonzero_limbs = bit_length.div_ceil(Limb::BITS) as usize;
    let partial_limb = bit_length % Limb::BITS;
    let mask = Word::MAX >> ((Word::BITS - partial_limb) % Word::BITS);

    for i in 0..nonzero_limbs - 1 {
        rng.try_fill_bytes(&mut buffer)
            .map_err(RandomBitsError::RandCore)?;
        zeroed_limbs[i] = Limb(Word::from_le_bytes(buffer));
    }

    // This algorithm should sample the same number of random bytes, regardless of the pointer width
    // of the target platform. To this end, special attention has to be paid to the case where
    // bit_length - 1 < 32 mod 64. Bit strings of that size can be represented using `2X+1` 32-bit
    // words or `X+1` 64-bit words. Note that 64*(X+1) - 32*(2X+1) = 32. Hence, if we sample full
    // words only, a 64-bit platform will sample 32 bits more than a 32-bit platform. We prevent
    // this by forcing both platforms to only sample 4 bytes for the last word in this case.
    let slice = if partial_limb > 0 && partial_limb <= 32 {
        // Note: we do not have to zeroize the second half of the buffer, as the mask will take
        // care of this in the end.
        &mut buffer[0..4]
    } else {
        buffer.as_mut_slice()
    };

    rng.try_fill_bytes(slice)
        .map_err(RandomBitsError::RandCore)?;
    zeroed_limbs[nonzero_limbs - 1] = Limb(Word::from_le_bytes(buffer) & mask);

    Ok(())
}

impl<const LIMBS: usize> RandomBits for Uint<LIMBS> {
    fn try_random_bits<R: TryRngCore + ?Sized>(
        rng: &mut R,
        bit_length: u32,
    ) -> Result<Self, RandomBitsError<R::Error>> {
        Self::try_random_bits_with_precision(rng, bit_length, Self::BITS)
    }

    fn try_random_bits_with_precision<R: TryRngCore + ?Sized>(
        rng: &mut R,
        bit_length: u32,
        bits_precision: u32,
    ) -> Result<Self, RandomBitsError<R::Error>> {
        if bits_precision != Self::BITS {
            return Err(RandomBitsError::BitsPrecisionMismatch {
                bits_precision,
                integer_bits: Self::BITS,
            });
        }
        if bit_length > Self::BITS {
            return Err(RandomBitsError::BitLengthTooLarge {
                bit_length,
                bits_precision,
            });
        }
        let mut limbs = [Limb::ZERO; LIMBS];
        random_bits_core(rng, &mut limbs, bit_length)?;
        Ok(Self::from(limbs))
    }
}

impl<const LIMBS: usize> RandomMod for Uint<LIMBS> {
    fn random_mod<R: RngCore + ?Sized>(rng: &mut R, modulus: &NonZero<Self>) -> Self {
        let mut n = Self::ZERO;
        let Ok(()) = random_mod_core(rng, &mut n, modulus, modulus.bits_vartime());
        n
    }

    fn try_random_mod<R: TryRngCore + ?Sized>(
        rng: &mut R,
        modulus: &NonZero<Self>,
    ) -> Result<Self, R::Error> {
        let mut n = Self::ZERO;
        random_mod_core(rng, &mut n, modulus, modulus.bits_vartime())?;
        Ok(n)
    }
}

/// Generic implementation of `random_mod` which can be shared with `BoxedUint`.
///
/// When combined with a platform-independent "8-byte sequential" `rng`, this function is
/// platform-independent. We consider an RNG "`X`-byte sequential" whenever
/// `rng.fill_bytes(&mut bytes[..i]); rng.fill_bytes(&mut bytes[i..])` constructs the same `bytes`,
/// as long as `i` is a multiple of `X`.
/// Note that the `TryRngCore` trait does _not_ require this behaviour from `rng`.
// TODO(tarcieri): obtain `n_bits` via a trait like `Integer`
pub(super) fn random_mod_core<T, R: TryRngCore + ?Sized>(
    rng: &mut R,
    n: &mut T,
    modulus: &NonZero<T>,
    n_bits: u32,
) -> Result<(), R::Error>
where
    T: AsMut<[Limb]> + AsRef<[Limb]> + ConstantTimeLess + Zero,
{
    // Always sample 8 bytes at a time for platform independence.
    // This matches the 64-bit algorithm behavior on all platforms.
    let mut next_u64 = || -> Result<u64, R::Error> {
        let mut bytes = [0u8; 8];
        rng.try_fill_bytes(&mut bytes)?;
        Ok(u64::from_le_bytes(bytes))
    };

    // Work in terms of 64-bit words for platform independence
    let n_words64 = n_bits.div_ceil(64) as usize;

    // Get high 64-bit word from modulus
    let hi_word64_modulus = get_u64_word(modulus.as_ref().as_ref(), n_words64 - 1);

    let mask64 = !0u64 >> hi_word64_modulus.leading_zeros();
    let mut hi_word64 = next_u64()? & mask64;

    loop {
        while hi_word64 > hi_word64_modulus {
            hi_word64 = next_u64()? & mask64;
        }
        // Set high 64-bit word
        set_u64_word(n.as_mut(), n_words64 - 1, hi_word64);
        // Set low 64-bit words
        for i in 0..n_words64 - 1 {
            let word = next_u64()?;
            set_u64_word(n.as_mut(), i, word);
        }
        // If the high word is equal to the modulus' high word, it's still possible
        // that the full uint is too big so we check and repeat if it is.
        if n.ct_lt(modulus).into() {
            break;
        }
        hi_word64 = next_u64()? & mask64;
    }
    Ok(())
}

/// Get the i-th 64-bit word from a limbs array (little-endian word order).
#[inline]
fn get_u64_word(limbs: &[Limb], i: usize) -> u64 {
    #[cfg(target_pointer_width = "64")]
    {
        limbs[i].0
    }
    #[cfg(target_pointer_width = "32")]
    {
        let lo = limbs[i * 2].0 as u64;
        let hi = limbs.get(i * 2 + 1).map_or(0, |l| l.0 as u64);
        lo | (hi << 32)
    }
}

/// Set the i-th 64-bit word in a limbs array (little-endian word order).
#[inline]
fn set_u64_word(limbs: &mut [Limb], i: usize, val: u64) {
    #[cfg(target_pointer_width = "64")]
    {
        limbs[i] = Limb(val);
    }
    #[cfg(target_pointer_width = "32")]
    {
        limbs[i * 2] = Limb(val as u32);
        if let Some(limb) = limbs.get_mut(i * 2 + 1) {
            *limb = Limb((val >> 32) as u32);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::uint::rand::{random_bits_core, random_mod_core};
    use crate::{Limb, NonZero, Random, RandomBits, RandomMod, U256, U1024, Uint};
    use chacha20::ChaCha8Rng;
    use rand_core::{RngCore, SeedableRng};

    const RANDOM_OUTPUT: U1024 = Uint::from_be_hex(concat![
        "A484C4C693EECC47C3B919AE0D16DF2259CD1A8A9B8EA8E0862878227D4B40A3",
        "C54302F2EB1E2F69E17653A37F1BCC44277FA208E6B31E08CDC4A23A7E88E660",
        "EF781C7DD2D368BAD438539D6A2E923C8CAE14CB947EB0BDE10D666732024679",
        "0F6760A48F9B887CB2FB0D3281E2A6E67746A55FBAD8C037B585F767A79A3B6C"
    ]);

    /// Construct a 4-sequential `rng`, i.e., an `rng` such that
    /// `rng.fill_bytes(&mut buffer[..x]); rng.fill_bytes(&mut buffer[x..])` will construct the
    /// same `buffer`, for `x` any in `0..buffer.len()` that is `0 mod 4`.
    fn get_four_sequential_rng() -> ChaCha8Rng {
        ChaCha8Rng::seed_from_u64(0)
    }

    /// Make sure the random value constructed is consistent across platforms
    #[test]
    fn random_platform_independence() {
        let mut rng = get_four_sequential_rng();
        assert_eq!(U1024::random(&mut rng), RANDOM_OUTPUT);
    }

    #[test]
    fn random_mod() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);

        // Ensure `random_mod` runs in a reasonable amount of time
        let modulus = NonZero::new(U256::from(42u8)).unwrap();
        let res = U256::random_mod(&mut rng, &modulus);

        // Check that the value is in range
        assert!(res < U256::from(42u8));

        // Ensure `random_mod` runs in a reasonable amount of time
        // when the modulus is larger than 1 limb
        let modulus = NonZero::new(U256::from(0x10000000000000001u128)).unwrap();
        let res = U256::random_mod(&mut rng, &modulus);

        // Check that the value is in range
        assert!(res < U256::from(0x10000000000000001u128));
    }

    #[test]
    fn random_bits() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);

        let lower_bound = 16;

        // Full length of the integer
        let bit_length = U256::BITS;
        for _ in 0..10 {
            let res = U256::random_bits(&mut rng, bit_length);
            assert!(res > (U256::ONE << (bit_length - lower_bound)));
        }

        // A multiple of limb size
        let bit_length = U256::BITS - Limb::BITS;
        for _ in 0..10 {
            let res = U256::random_bits(&mut rng, bit_length);
            assert!(res > (U256::ONE << (bit_length - lower_bound)));
            assert!(res < (U256::ONE << bit_length));
        }

        // A multiple of 8
        let bit_length = U256::BITS - Limb::BITS - 8;
        for _ in 0..10 {
            let res = U256::random_bits(&mut rng, bit_length);
            assert!(res > (U256::ONE << (bit_length - lower_bound)));
            assert!(res < (U256::ONE << bit_length));
        }

        // Not a multiple of 8
        let bit_length = U256::BITS - Limb::BITS - 8 - 3;
        for _ in 0..10 {
            let res = U256::random_bits(&mut rng, bit_length);
            assert!(res > (U256::ONE << (bit_length - lower_bound)));
            assert!(res < (U256::ONE << bit_length));
        }

        // One incomplete limb
        let bit_length = 7;
        for _ in 0..10 {
            let res = U256::random_bits(&mut rng, bit_length);
            assert!(res < (U256::ONE << bit_length));
        }

        // Zero bits
        let bit_length = 0;
        for _ in 0..10 {
            let res = U256::random_bits(&mut rng, bit_length);
            assert_eq!(res, U256::ZERO);
        }
    }

    /// Make sure the random_bits output is consistent across platforms
    #[test]
    fn random_bits_platform_independence() {
        let mut rng = get_four_sequential_rng();

        let bit_length = 989;
        let mut val = U1024::ZERO;
        random_bits_core(&mut rng, val.as_mut_limbs(), bit_length).expect("safe");

        assert_eq!(
            val,
            RANDOM_OUTPUT.bitand(&U1024::ONE.shl(bit_length).wrapping_sub(&Uint::ONE))
        );

        // Test that the RNG is in the same state
        let mut state = [0u8; 16];
        rng.fill_bytes(&mut state);

        assert_eq!(
            state,
            [
                198, 196, 132, 164, 240, 211, 223, 12, 36, 189, 139, 48, 94, 1, 123, 253
            ]
        );
    }

    /// Test that random bytes are sampled consecutively.
    #[test]
    fn random_bits_4_bytes_sequential() {
        // Test for multiples of 4 bytes, i.e., multiples of 32 bits.
        let bit_lengths = [0, 32, 64, 128, 192, 992];

        for bit_length in bit_lengths {
            let mut rng = get_four_sequential_rng();
            let mut first = U1024::ZERO;
            let mut second = U1024::ZERO;
            random_bits_core(&mut rng, first.as_mut_limbs(), bit_length).expect("safe");
            random_bits_core(&mut rng, second.as_mut_limbs(), U1024::BITS - bit_length)
                .expect("safe");
            assert_eq!(second.shl(bit_length).bitor(&first), RANDOM_OUTPUT);
        }
    }

    /// Make sure the random_mod output is consistent across platforms.
    ///
    /// The algorithm samples high word first, then low words in order.
    /// With MAX modulus (no rejection), this rearranges RANDOM_OUTPUT words:
    /// - output[15] gets RNG bytes 0-7 (= RANDOM_OUTPUT[0])
    /// - output[0] gets RNG bytes 8-15 (= RANDOM_OUTPUT[1])
    /// - output[i] gets RNG bytes (i+1)*8..(i+2)*8 for i in 0..15
    /// - output[14] gets RNG bytes 120-127 (= RANDOM_OUTPUT[15])
    #[test]
    fn random_mod_platform_independence() {
        let mut rng = get_four_sequential_rng();

        // Use MAX modulus so there's no rejection (simplifies expected value calculation)
        let modulus = NonZero::new(U1024::MAX).unwrap();

        let mut result = U1024::ZERO;
        random_mod_core(&mut rng, &mut result, &modulus, U1024::BITS).expect("safe");

        // Expected value: rearrangement of RANDOM_OUTPUT words as described above
        // RANDOM_OUTPUT[0] -> result[15] (high word)
        // RANDOM_OUTPUT[1] -> result[0]
        // RANDOM_OUTPUT[2] -> result[1]
        // ...
        // RANDOM_OUTPUT[15] -> result[14]
        let expected = Uint::from_be_hex(concat![
            "B585F767A79A3B6CA484C4C693EECC47C3B919AE0D16DF2259CD1A8A9B8EA8E0",
            "862878227D4B40A3C54302F2EB1E2F69E17653A37F1BCC44277FA208E6B31E08",
            "CDC4A23A7E88E660EF781C7DD2D368BAD438539D6A2E923C8CAE14CB947EB0BD",
            "E10D6667320246790F6760A48F9B887CB2FB0D3281E2A6E67746A55FBAD8C037"
        ]);

        assert_eq!(result, expected);

        // Test that the RNG is in the same state after consuming 128 bytes
        let mut state = [0u8; 16];
        rng.fill_bytes(&mut state);

        assert_eq!(
            state,
            [
                240, 211, 223, 12, 36, 189, 139, 48, 94, 1, 123, 253, 193, 64, 74, 110
            ]
        );
    }
}
