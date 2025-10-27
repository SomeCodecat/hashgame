// Integration test that uses the crate's small library API.
use rust_miner::add;

#[test]
fn integration_adds() {
    assert_eq!(add(1, 2), 3);
}
