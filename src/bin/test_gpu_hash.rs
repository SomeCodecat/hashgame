use ocl::{Platform, Context, Device, Kernel, Program, Queue, Buffer, flags};

fn main() {
    // Test with realistic parent hash
    let test_parent = "9c49c67671a8e2bd3fa0fc400b44ff";  // Simulated parent (partial)
    let test_name_with_spaces = " TEST-B6 ";
    let test_seed_num: u64 = 0;
    
    // Build message the same way GPU does
    let mut msg_bytes = Vec::new();
    msg_bytes.extend_from_slice(test_parent.as_bytes());
    msg_bytes.extend_from_slice(test_name_with_spaces.as_bytes());
    msg_bytes.extend_from_slice(test_seed_num.to_string().as_bytes());
    
    let test_msg = String::from_utf8(msg_bytes.clone()).unwrap();
    
    println!("Testing GPU SHA-256 for message: '{}'", test_msg);
    println!("Message length: {} bytes", msg_bytes.len());
    println!();
    
    // Setup OpenCL
    let platform = Platform::default();
    let device = Device::first(platform).unwrap();
    let context = Context::builder().devices(device).build().unwrap();
    let queue = Queue::new(&context, device, None).unwrap();
    
    // Simplified kernel that just computes one hash and returns it
    let src = r#"
    #define rotr(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
    #define ch(x, y, z) (((x) & (y)) ^ (~(x) & (z)))
    #define maj(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
    #define big_sigma0(x) (rotr(x,2) ^ rotr(x,13) ^ rotr(x,22))
    #define big_sigma1(x) (rotr(x,6) ^ rotr(x,11) ^ rotr(x,25))
    #define small_sigma0(x) (rotr(x,7) ^ rotr(x,18) ^ ((x) >> 3))
    #define small_sigma1(x) (rotr(x,17) ^ rotr(x,19) ^ ((x) >> 10))

    __constant uint K[64] = {
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
    };

    __kernel void test_hash(__global const uchar* msg_bytes, uint msg_len, __global uint* hash_out) {
        uint w[64];
        
        // Initialize all words to zero
        for (uint i = 0; i < 16; i++) {
            w[i] = 0;
        }
        
        // Load message bytes into words (big-endian)
        for (uint i = 0; i < msg_len; i++) {
            uint word_idx = i / 4;
            uint byte_pos = i % 4;
            w[word_idx] |= ((uint)msg_bytes[i]) << (24 - byte_pos * 8);
        }
        
        // Add padding
        uint pad_idx = msg_len / 4;
        uint pad_pos = msg_len % 4;
        w[pad_idx] |= 0x80 << (24 - pad_pos * 8);
        
        // Set length
        w[14] = 0;
        w[15] = msg_len * 8;
        
        // Extend to 64 words
        for (uint i = 16; i < 64; i++) {
            w[i] = small_sigma1(w[i-2]) + w[i-7] + small_sigma0(w[i-15]) + w[i-16];
        }
        
        // Initialize
        uint h0 = 0x6a09e667, h1 = 0xbb67ae85, h2 = 0x3c6ef372, h3 = 0xa54ff53a;
        uint h4 = 0x510e527f, h5 = 0x9b05688c, h6 = 0x1f83d9ab, h7 = 0x5be0cd19;
        uint a = h0, b = h1, c = h2, d = h3, e = h4, f = h5, g = h6, h = h7;
        
        // Compression
        for (uint i = 0; i < 64; i++) {
            uint T1 = h + big_sigma1(e) + ch(e,f,g) + K[i] + w[i];
            uint T2 = big_sigma0(a) + maj(a,b,c);
            h = g; g = f; f = e; e = d + T1;
            d = c; c = b; b = a; a = T1 + T2;
        }
        
        // Output hash
        hash_out[0] = h0 + a;
        hash_out[1] = h1 + b;
        hash_out[2] = h2 + c;
        hash_out[3] = h3 + d;
        hash_out[4] = h4 + e;
        hash_out[5] = h5 + f;
        hash_out[6] = h6 + g;
        hash_out[7] = h7 + h;
    }
    "#;
    
    let program = Program::builder().src(src).build(&context).unwrap();
    
    // Prepare buffers
    let msg_bytes = test_msg.as_bytes();
    let msg_buf: Buffer<u8> = Buffer::builder()
        .queue(queue.clone())
        .flags(flags::MEM_READ_ONLY | flags::MEM_COPY_HOST_PTR)
        .len(msg_bytes.len())
        .copy_host_slice(msg_bytes)
        .build().unwrap();
    
    let hash_buf: Buffer<u32> = Buffer::builder()
        .queue(queue.clone())
        .flags(flags::MEM_WRITE_ONLY)
        .len(8)
        .build().unwrap();
    
    // Run kernel
    let kernel = Kernel::builder()
        .program(&program)
        .name("test_hash")
        .queue(queue.clone())
        .global_work_size(1)
        .arg(&msg_buf)
        .arg(msg_bytes.len() as u32)
        .arg(&hash_buf)
        .build().unwrap();
    
    unsafe { kernel.enq().unwrap(); }
    
    // Read result
    let mut hash = vec![0u32; 8];
    hash_buf.read(&mut hash).enq().unwrap();
    queue.finish().unwrap();
    
    println!("GPU hash (as u32 words): {:08x?}", hash);
    
    // Convert to bytes (big-endian)
    let mut hash_bytes = vec![];
    for word in &hash {
        hash_bytes.push((word >> 24) as u8);
        hash_bytes.push((word >> 16) as u8);
        hash_bytes.push((word >> 8) as u8);
        hash_bytes.push(*word as u8);
    }
    
    print!("GPU hash (as hex bytes): ");
    for byte in &hash_bytes {
        print!("{:02x}", byte);
    }
    println!();
    
    println!("\nMatch: {}", hash_bytes == vec![0xc9, 0x8d, 0x6a, 0x70, 0x1e, 0x4b, 0x5c, 0xb1, 0xdc, 0xa8, 0xc9, 0xf2, 0xee, 0xdf, 0xc3, 0xe8, 0x08, 0x9a, 0x2d, 0xd9, 0x3e, 0x46, 0x0d, 0x25, 0xf6, 0x76, 0x48, 0xce, 0x94, 0xc3, 0x78, 0xe6]);
}
