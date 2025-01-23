#[allow(warnings)]
#[rustfmt::skip]
mod bindings;

struct Component;

bindings::export!(Component with_types_in bindings);
