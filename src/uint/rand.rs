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

    /// Diagnostic test: Check borrowing_sub specifically
    /// This might reveal issues with u128 arithmetic under QEMU
    #[test]
    fn borrowing_sub_diagnostic() {
        // Test: 55 - 8192 should produce a borrow
        let val_55 = U256::from_u32(55);
        let val_8192 = U256::from_u32(8192);

        let (_result, borrow) = val_55.borrowing_sub(&val_8192, Limb::ZERO);

        // The result should be 55 - 8192 wrapped (a very large number)
        // The borrow should be non-zero (indicating underflow)
        assert!(
            borrow.0 != 0,
            "borrowing_sub(55, 8192) should produce non-zero borrow, got borrow={}",
            borrow.0
        );

        // Test: 8192 - 55 should NOT produce a borrow
        let (result2, borrow2) = val_8192.borrowing_sub(&val_55, Limb::ZERO);

        assert_eq!(
            borrow2.0, 0,
            "borrowing_sub(8192, 55) should produce zero borrow, got borrow={}",
            borrow2.0
        );
        assert_eq!(result2, U256::from_u32(8192 - 55));
    }

    /// Diagnostic test: Check repeated calls to random_bits_core
    /// This tests if there's an issue with non-zeroed buffers
    #[test]
    fn random_bits_core_repeated_calls() {
        let mut rng = get_four_sequential_rng();
        let mut n = U256::ZERO;

        // First call - should produce a 14-bit value
        random_bits_core(&mut rng, n.as_mut_limbs(), 14).expect("safe");
        let first_val = n.as_limbs()[0].0;
        assert!(first_val < 16384, "First value {} should be < 16384", first_val);

        // Check that high limbs are still zero
        assert_eq!(n.as_limbs()[1].0, 0, "Limb 1 should be zero");
        assert_eq!(n.as_limbs()[2].0, 0, "Limb 2 should be zero");
        assert_eq!(n.as_limbs()[3].0, 0, "Limb 3 should be zero");

        // Second call - with non-zero first limb (simulating rejection loop)
        random_bits_core(&mut rng, n.as_mut_limbs(), 14).expect("safe");
        let second_val = n.as_limbs()[0].0;
        assert!(second_val < 16384, "Second value {} should be < 16384", second_val);

        // High limbs should still be zero
        assert_eq!(n.as_limbs()[1].0, 0, "Limb 1 should still be zero after second call");
        assert_eq!(n.as_limbs()[2].0, 0, "Limb 2 should still be zero after second call");
        assert_eq!(n.as_limbs()[3].0, 0, "Limb 3 should still be zero after second call");

        // Values should be different (with overwhelming probability)
        assert_ne!(first_val, second_val, "Values should differ between calls");
    }

    /// Diagnostic: Compare native > comparison with ct_lt
    /// This tests if the issue is specifically in ct_lt / borrowing_sub
    #[test]
    fn ct_lt_vs_native_comparison() {
        use subtle::ConstantTimeLess;

        // Test a range of values to see if ct_lt matches native comparison
        let modulus = U256::from_u32(8192);

        // Values that should be LESS than 8192 (ct_lt should return true)
        let less_values = [0u32, 1, 55, 100, 1000, 8191];
        for v in less_values {
            let val = U256::from_u32(v);
            let ct_result: bool = val.ct_lt(&modulus).into();
            let native_result = v < 8192;
            assert_eq!(
                ct_result, native_result,
                "Mismatch for value {}: ct_lt={}, native={}",
                v, ct_result, native_result
            );
        }

        // Values that should be GREATER OR EQUAL to 8192 (ct_lt should return false)
        let ge_values = [8192u32, 8193, 10000, 16383, 65535];
        for v in ge_values {
            let val = U256::from_u32(v);
            let ct_result: bool = val.ct_lt(&modulus).into();
            let native_result = v < 8192;
            assert_eq!(
                ct_result, native_result,
                "Mismatch for value {}: ct_lt={}, native={}",
                v, ct_result, native_result
            );
        }
    }

    /// Diagnostic: Test the full random_mod_core with minimal iterations
    /// This isolates the issue by limiting the loop iterations
    #[test]
    fn random_mod_core_bounded_loop_diagnostic() {
        extern crate std;
        use std::eprintln;
        use subtle::ConstantTimeLess;

        let mut rng = get_four_sequential_rng();
        let modulus = NonZero::new(U256::from_u32(8192)).unwrap();
        let n_bits = modulus.bits_vartime();

        // Should be 14 bits for 8192
        assert_eq!(n_bits, 14, "bits_vartime for 8192 should be 14");

        let mut n = U256::ZERO;
        let mut iterations = 0;
        const MAX_ITERATIONS: usize = 100;

        loop {
            random_bits_core(&mut rng, n.as_mut_limbs(), n_bits).expect("safe");

            let n_val = n.as_limbs()[0].0;
            let modulus_inner: &U256 = &modulus;
            let is_less: bool = n.ct_lt(modulus_inner).into();

            eprintln!(
                "iter={} n={} n<8192={} ct_lt={}",
                iterations, n_val, n_val < 8192, is_less
            );

            if is_less {
                eprintln!("SUCCESS: Found valid value {} at iteration {}", n_val, iterations);
                break;
            }

            iterations += 1;
            if iterations >= MAX_ITERATIONS {
                panic!(
                    "FAILED: Loop did not terminate after {} iterations. Last n={}",
                    MAX_ITERATIONS, n_val
                );
            }
        }

        // Verify the final value is correct
        assert!(
            n < U256::from_u32(8192),
            "Final value {} should be < 8192",
            n.as_limbs()[0].0
        );
    }

    /// Diagnostic: Test u128 arithmetic that borrowing_sub relies on
    /// This specifically tests the pattern used in primitives::borrowing_sub
    #[test]
    fn u128_subtraction_borrow_diagnostic() {
        // Reproduce the exact u128 arithmetic from borrowing_sub
        let a: u128 = 55;  // lhs as WideWord
        let b: u128 = 8192;  // rhs as WideWord
        let ret = a.wrapping_sub(b);

        // Extract low and high words
        let low = ret as u64;
        let high = (ret >> 64) as u64;

        // When a < b, ret should be 2^128 + (a - b) = 2^128 - 8137
        // low should be 2^64 - 8137 = 0xFFFFFFFFFFFFE017
        // high should be 2^64 - 1 = 0xFFFFFFFFFFFFFFFF
        assert_eq!(
            high, u64::MAX,
            "u128 subtraction borrow should produce high word = MAX, got {:#018x}",
            high
        );

        // Also verify the low word
        let expected_low = 0u64.wrapping_sub(8137);  // 2^64 - 8137
        assert_eq!(
            low, expected_low,
            "u128 subtraction should produce expected low word, got {:#018x} expected {:#018x}",
            low, expected_low
        );

        // Test the specific pattern from borrowing_sub
        // This matches: let ret = a.wrapping_sub(b + borrow);
        let borrow_input: u64 = 0;
        let borrow = (borrow_input >> 63) as u128;  // Should be 0
        let ret2 = a.wrapping_sub(b + borrow);
        let computed_borrow = (ret2 >> 64) as u64;
        assert_eq!(
            computed_borrow, u64::MAX,
            "borrowing_sub pattern should produce borrow = MAX, got {:#018x}",
            computed_borrow
        );
    }

    /// Diagnostic: Compare fill_bytes vs next_u64 byte sequences
    /// This tests if the RNG produces the same bytes regardless of how we consume them
    #[test]
    fn rng_fill_bytes_vs_next_u64() {
        use rand_core::RngCore;

        // Test with fill_bytes
        let mut rng1 = get_four_sequential_rng();
        let mut buf = [0u8; 8];
        rng1.fill_bytes(&mut buf);
        let fill_bytes_result = u64::from_le_bytes(buf);

        // Test with next_u64
        let mut rng2 = get_four_sequential_rng();
        let next_u64_result = rng2.next_u64();

        assert_eq!(
            fill_bytes_result, next_u64_result,
            "fill_bytes ({:#018x}) and next_u64 ({:#018x}) should produce same value",
            fill_bytes_result, next_u64_result
        );

        // Test a few more to be sure
        for _ in 0..10 {
            rng1.fill_bytes(&mut buf);
            let fb = u64::from_le_bytes(buf);
            let nu = rng2.next_u64();
            assert_eq!(fb, nu, "fill_bytes and next_u64 diverged");
        }
    }

    /// Diagnostic: Compare 4-byte fill_bytes sequences
    /// The new algorithm uses 4-byte chunks for partial limbs
    #[test]
    fn rng_fill_4_bytes_consistency() {
        use rand_core::RngCore;

        // Fill 8 bytes in two 4-byte chunks
        let mut rng1 = get_four_sequential_rng();
        let mut buf1 = [0u8; 4];
        let mut buf2 = [0u8; 4];
        rng1.fill_bytes(&mut buf1);
        rng1.fill_bytes(&mut buf2);

        // Fill 8 bytes in one chunk
        let mut rng2 = get_four_sequential_rng();
        let mut buf_full = [0u8; 8];
        rng2.fill_bytes(&mut buf_full);

        // They should be the same (4-byte sequential property of ChaCha)
        assert_eq!(
            &buf1[..], &buf_full[0..4],
            "First 4 bytes should match"
        );
        assert_eq!(
            &buf2[..], &buf_full[4..8],
            "Second 4 bytes should match"
        );
    }

    /// Diagnostic: Test the old algorithm's pre-filter logic
    /// This simulates what the OLD random_mod_core does
    #[test]
    fn old_algorithm_prefilter_simulation() {
        let mut rng = get_four_sequential_rng();

        let hi_word_modulus: u64 = 8192;
        let mask: u64 = !0 >> hi_word_modulus.leading_zeros();

        // Generate a value using the OLD algorithm's approach
        let mut hi_word = rng.next_u64() & mask;

        // Pre-filter: reject if > hi_word_modulus (uses native comparison)
        let mut iterations = 0;
        while hi_word > hi_word_modulus {
            hi_word = rng.next_u64() & mask;
            iterations += 1;
            assert!(iterations < 1000, "Pre-filter loop running too long");
        }

        // After pre-filter, hi_word <= 8192
        assert!(
            hi_word <= 8192,
            "After pre-filter, hi_word={} should be <= 8192",
            hi_word
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
