#[allow(dead_code)]
pub mod exports {
    #[allow(dead_code)]
    pub mod augurs {
        #[allow(dead_code)]
        pub mod ets {
            #[allow(dead_code, clippy::all)]
            pub mod ets {
                #[used]
                #[doc(hidden)]
                static __FORCE_SECTION_REF: fn() = super::super::super::super::__link_custom_section_describing_imports;
                #[doc(hidden)]
                macro_rules! __export_augurs_ets_ets_cabi {
                    ($ty:ident with_types_in $($path_to_types:tt)*) => {
                        const _ : () = {};
                    };
                }
                #[doc(hidden)]
                pub(crate) use __export_augurs_ets_ets_cabi;
            }
        }
    }
}
/// Generates `#[no_mangle]` functions to export the specified type as the
/// root implementation of all generated traits.
///
/// For more information see the documentation of `wit_bindgen::generate!`.
///
/// ```rust
/// # macro_rules! export{ ($($t:tt)*) => (); }
/// # trait Guest {}
/// struct MyType;
///
/// impl Guest for MyType {
///     // ...
/// }
///
/// export!(MyType);
/// ```
#[allow(unused_macros)]
#[doc(hidden)]
macro_rules! __export_component_impl {
    ($ty:ident) => {
        self::export!($ty with_types_in self);
    };
    ($ty:ident with_types_in $($path_to_types_root:tt)*) => {
        $($path_to_types_root)*::
        exports::augurs::ets::ets::__export_augurs_ets_ets_cabi!($ty with_types_in
        $($path_to_types_root)*:: exports::augurs::ets::ets);
    };
}
#[doc(inline)]
pub(crate) use __export_component_impl as export;
#[cfg(target_arch = "wasm32")]
#[link_section = "component-type:wit-bindgen:0.35.0:augurs:ets:component:encoded world"]
#[doc(hidden)]
pub static __WIT_BINDGEN_COMPONENT_TYPE: [u8; 180] = *b"\
\0asm\x0d\0\x01\0\0\x19\x16wit-component-encoding\x04\0\x075\x01A\x02\x01A\x02\x01\
B\0\x04\0\x0eaugurs:ets/ets\x05\0\x04\0\x14augurs:ets/component\x04\0\x0b\x0f\x01\
\0\x09component\x03\0\0\0G\x09producers\x01\x0cprocessed-by\x02\x0dwit-component\
\x070.220.0\x10wit-bindgen-rust\x060.35.0";
#[inline(never)]
#[doc(hidden)]
pub fn __link_custom_section_describing_imports() {
    wit_bindgen_rt::maybe_link_cabi_realloc();
}
