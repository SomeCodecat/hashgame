use sha2::{Digest, Sha256};

fn main() {
    // Test message matching GPU test
    let message = "9c49c67671a8e2bd3fa0fc400b44ff TEST-B6 0";
    let mut hasher = Sha256::new();
    hasher.update(message.as_bytes());
    let hash = hasher.finalize();
    
    println!("Message: '{}'", message);
    println!("Message length: {} bytes", message.len());
    println!("\nHash as u32 words (big-endian):");
    for i in 0..8 {
        let word = u32::from_be_bytes([hash[i*4], hash[i*4+1], hash[i*4+2], hash[i*4+3]]);
        println!("  word[{}] = 0x{:08x}", i, word);
    }
    
    // Count leading zero bits
    let mut count = 0;
    for &byte in hash.iter() {
        if byte == 0 {
            count += 8;
        } else {
            count += byte.leading_zeros() as usize;
            break;
        }
    }
    println!("\nLeading zero bits: {}", count);
}
