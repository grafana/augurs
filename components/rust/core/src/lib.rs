#[allow(warnings)]
#[rustfmt::skip]
mod bindings;

use bindings::exports::augurs::clustering::dbscan::{Guest, GuestDbscan, Options as DbscanOptions};

struct Component;

impl Guest for Component {
    type Dbscan = Dbscan;
}

struct Dbscan;

impl GuestDbscan for Dbscan {
    fn new(options: DbscanOptions) -> Self {
        todo!()
    }
}

bindings::export!(Component with_types_in bindings);
