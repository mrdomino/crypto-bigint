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
) -> Result<(), R::Error> {
    if bit_length == 0 {
        return Ok(());
    }

    let buffer: Word = 0;
    let mut buffer = buffer.to_be_bytes();

    let nonzero_limbs = bit_length.div_ceil(Limb::BITS) as usize;
    let partial_limb = bit_length % Limb::BITS;
    let mask = Word::MAX >> ((Word::BITS - partial_limb) % Word::BITS);

    for i in 0..nonzero_limbs - 1 {
        rng.try_fill_bytes(&mut buffer)?;
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

    rng.try_fill_bytes(slice)?;
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
        random_bits_core(rng, &mut limbs, bit_length).map_err(RandomBitsError::RandCore)?;
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
    loop {
        random_bits_core(rng, n.as_mut(), n_bits)?;

        if n.ct_lt(modulus).into() {
            break;
        }
    }
    Ok(())
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

    /// Make sure random_mod output is consistent across platforms
    #[test]
    fn random_mod_platform_independence() {
        let mut rng = get_four_sequential_rng();

        let modulus = NonZero::new(U256::from_u32(8192)).unwrap();
        let mut vals = [U256::ZERO, U256::ZERO, U256::ZERO, U256::ZERO, U256::ZERO];
        for val in &mut vals {
            random_mod_core(&mut rng, val, &modulus, modulus.bits_vartime()).unwrap();
        }
        let expected = [55, 3378, 2172, 1657, 5323];
        for (want, got) in expected.into_iter().zip(vals.into_iter()) {
            assert_eq!(got, U256::from_u32(want));
        }

        let mut state = [0u8; 16];
        rng.fill_bytes(&mut state);

        assert_eq!(
            state,
            [
                60, 146, 46, 106, 157, 83, 56, 212, 186, 104, 211, 210, 125, 28, 120, 239
            ],
        );
    }

    /// Diagnostic test: Check what random_bits_core produces for small bit lengths
    /// This helps identify if the issue is in random_bits_core itself
    #[test]
    fn random_bits_core_small_bits_diagnostic() {
        let mut rng = get_four_sequential_rng();

        // Test 14-bit generation (same as the modulus test uses)
        let mut val = U256::ZERO;
        random_bits_core(&mut rng, val.as_mut_limbs(), 14).expect("safe");

        // The value should be at most 14 bits (< 16384)
        assert!(val < U256::from_u32(16384), "14-bit value should be < 16384, got {:?}", val);

        // Generate several more and verify they're all in range
        for _ in 0..10 {
            let mut val = U256::ZERO;
            random_bits_core(&mut rng, val.as_mut_limbs(), 14).expect("safe");
            assert!(val < U256::from_u32(16384), "14-bit value should be < 16384");
        }
    }

    /// Diagnostic test: Use 2^14-1 modulus (minimal rejection)
    /// If this hangs, the issue is NOT in rejection sampling
    #[test]
    fn random_mod_minimal_rejection() {
        let mut rng = get_four_sequential_rng();

        // Use 16383 = 2^14 - 1, so almost no rejection (only 1/16384 chance)
        // bits_vartime(16383) = 14, so we generate 14-bit values
        let modulus = NonZero::new(U256::from_u32(16383)).unwrap();

        // Verify bits_vartime returns 14
        assert_eq!(modulus.bits_vartime(), 14);

        // This should complete with minimal rejection
        let mut val = U256::ZERO;
        random_mod_core(&mut rng, &mut val, &modulus, modulus.bits_vartime()).unwrap();

        // Value should be < 16383
        assert!(val < U256::from_u32(16383));
    }

    /// Diagnostic test: Verify ct_lt comparison works correctly for small values
    #[test]
    fn ct_lt_small_values_diagnostic() {
        use subtle::ConstantTimeLess;

        // Test values around 8192
        let modulus = U256::from_u32(8192);

        // Values less than 8192 should return true
        assert!(bool::from(U256::from_u32(0).ct_lt(&modulus)), "0 < 8192");
        assert!(bool::from(U256::from_u32(55).ct_lt(&modulus)), "55 < 8192");
        assert!(bool::from(U256::from_u32(8191).ct_lt(&modulus)), "8191 < 8192");

        // Values >= 8192 should return false
        assert!(!bool::from(U256::from_u32(8192).ct_lt(&modulus)), "8192 !< 8192");
        assert!(!bool::from(U256::from_u32(8193).ct_lt(&modulus)), "8193 !< 8192");
        assert!(!bool::from(U256::from_u32(16383).ct_lt(&modulus)), "16383 !< 8192");
    }

    /// Diagnostic test: Manually step through random_mod_core logic
    #[test]
    fn random_mod_manual_step_diagnostic() {
        use subtle::ConstantTimeLess;

        let mut rng = get_four_sequential_rng();
        let modulus = NonZero::new(U256::from_u32(8192)).unwrap();
        let n_bits = modulus.bits_vartime();

        // Should be 14 bits for 8192
        assert_eq!(n_bits, 14, "bits_vartime for 8192 should be 14");

        // Generate a value using random_bits_core
        let mut n = U256::ZERO;
        random_bits_core(&mut rng, n.as_mut_limbs(), n_bits).expect("safe");

        // Check the generated value
        let n_u64 = n.as_limbs()[0].0;
        assert!(n_u64 < 16384, "Generated value {} should be < 16384 (14 bits)", n_u64);

        // Check comparison (compare against the inner value, not NonZero wrapper)
        let modulus_inner: &U256 = &modulus;
        let is_less = n.ct_lt(modulus_inner);
        let is_less_bool: bool = is_less.into();

        // If n < 8192, is_less should be true
        if n_u64 < 8192 {
            assert!(is_less_bool, "n={} < 8192, but ct_lt returned false", n_u64);
        } else {
            assert!(!is_less_bool, "n={} >= 8192, but ct_lt returned true", n_u64);
        }
    }

    /// Diagnostic test: Detailed debug info for CI failure analysis
    /// This test prints values to help debug the cross/QEMU hang
    #[test]
    fn random_mod_detailed_debug() {
        extern crate std;
        use std::eprintln;
        use subtle::ConstantTimeLess;

        let mut rng = get_four_sequential_rng();

        // First, verify the raw bytes from the RNG
        let mut raw_bytes = [0u8; 8];
        rng.fill_bytes(&mut raw_bytes);
        eprintln!("First 8 raw bytes from ChaCha8: {:?}", raw_bytes);

        // Reset RNG
        let mut rng = get_four_sequential_rng();

        // Test random_bits_core with 14 bits
        let mut n = U256::ZERO;
        random_bits_core(&mut rng, n.as_mut_limbs(), 14).expect("safe");
        eprintln!("After random_bits_core(14): limbs = {:?}", n.as_limbs());
        eprintln!("First limb value: {}", n.as_limbs()[0].0);

        // Reset RNG again for the full test
        let mut rng = get_four_sequential_rng();
        let modulus = NonZero::new(U256::from_u32(8192)).unwrap();

        // Test with bounded iteration count to prevent hang
        let mut iteration = 0;
        let max_iterations = 100;
        let mut n = U256::ZERO;

        while iteration < max_iterations {
            random_bits_core(&mut rng, n.as_mut_limbs(), 14).expect("safe");
            let n_val = n.as_limbs()[0].0;

            let modulus_inner: &U256 = &modulus;
            let is_less = n.ct_lt(modulus_inner);
            let is_less_bool: bool = is_less.into();

            eprintln!(
                "Iteration {}: n={}, is_less={}, expected_is_less={}",
                iteration, n_val, is_less_bool, n_val < 8192
            );

            if is_less_bool {
                eprintln!("Breaking at iteration {} with n={}", iteration, n_val);
                break;
            }

            iteration += 1;
        }

        assert!(
            iteration < max_iterations,
            "Loop did not terminate after {} iterations. Last n={}",
            max_iterations,
            n.as_limbs()[0].0
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
}
