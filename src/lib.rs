// Small library helpers so `cargo test` can exercise a tiny, safe piece of code
// without invoking the heavy network/OpenCL mining binary.

/// Adds two integers. Used by the minimal tests.
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_adds() {
        assert_eq!(add(2, 3), 5);
    }
}
