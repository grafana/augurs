#[allow(dead_code)]
pub mod augurs {
    #[allow(dead_code)]
    pub mod core {
        #[allow(dead_code, clippy::all)]
        pub mod types {
            #[used]
            #[doc(hidden)]
            static __FORCE_SECTION_REF: fn() = super::super::super::__link_custom_section_describing_imports;
            use super::super::super::_rt;
            /// A sequence of floats comprising a time series.
            pub type TimeSeries = _rt::Vec<f64>;
            /// A matrix of pairwise distances between time series.
            pub type DistanceMatrix = _rt::Vec<_rt::Vec<f64>>;
        }
    }
}
#[allow(dead_code)]
pub mod exports {
    #[allow(dead_code)]
    pub mod augurs {
        #[allow(dead_code)]
        pub mod dtw {
            #[allow(dead_code, clippy::all)]
            pub mod dtw {
                #[used]
                #[doc(hidden)]
                static __FORCE_SECTION_REF: fn() = super::super::super::super::__link_custom_section_describing_imports;
                use super::super::super::super::_rt;
                pub type TimeSeries = super::super::super::super::augurs::core::types::TimeSeries;
                pub type DistanceMatrix = super::super::super::super::augurs::core::types::DistanceMatrix;
                /// Options for the DTW algorithm.
                #[repr(C)]
                #[derive(Clone, Copy)]
                pub struct DtwOpts {
                    /// The size of the Sakoe-Chiba warping window, `w`.
                    ///
                    /// Using a window limits shifts up to this amount away from the diagonal.
                    pub window: Option<u32>,
                    pub max_distance: Option<f64>,
                    pub lower_bound: Option<f64>,
                    pub upper_bound: Option<f64>,
                }
                impl ::core::fmt::Debug for DtwOpts {
                    fn fmt(
                        &self,
                        f: &mut ::core::fmt::Formatter<'_>,
                    ) -> ::core::fmt::Result {
                        f.debug_struct("DtwOpts")
                            .field("window", &self.window)
                            .field("max-distance", &self.max_distance)
                            .field("lower-bound", &self.lower_bound)
                            .field("upper-bound", &self.upper_bound)
                            .finish()
                    }
                }
                /// A Dynamic Time Warping calculator.
                #[derive(Debug)]
                #[repr(transparent)]
                pub struct Dtw {
                    handle: _rt::Resource<Dtw>,
                }
                type _DtwRep<T> = Option<T>;
                impl Dtw {
                    /// Creates a new resource from the specified representation.
                    ///
                    /// This function will create a new resource handle by moving `val` onto
                    /// the heap and then passing that heap pointer to the component model to
                    /// create a handle. The owned handle is then returned as `Dtw`.
                    pub fn new<T: GuestDtw>(val: T) -> Self {
                        Self::type_guard::<T>();
                        let val: _DtwRep<T> = Some(val);
                        let ptr: *mut _DtwRep<T> = _rt::Box::into_raw(
                            _rt::Box::new(val),
                        );
                        unsafe { Self::from_handle(T::_resource_new(ptr.cast())) }
                    }
                    /// Gets access to the underlying `T` which represents this resource.
                    pub fn get<T: GuestDtw>(&self) -> &T {
                        let ptr = unsafe { &*self.as_ptr::<T>() };
                        ptr.as_ref().unwrap()
                    }
                    /// Gets mutable access to the underlying `T` which represents this
                    /// resource.
                    pub fn get_mut<T: GuestDtw>(&mut self) -> &mut T {
                        let ptr = unsafe { &mut *self.as_ptr::<T>() };
                        ptr.as_mut().unwrap()
                    }
                    /// Consumes this resource and returns the underlying `T`.
                    pub fn into_inner<T: GuestDtw>(self) -> T {
                        let ptr = unsafe { &mut *self.as_ptr::<T>() };
                        ptr.take().unwrap()
                    }
                    #[doc(hidden)]
                    pub unsafe fn from_handle(handle: u32) -> Self {
                        Self {
                            handle: _rt::Resource::from_handle(handle),
                        }
                    }
                    #[doc(hidden)]
                    pub fn take_handle(&self) -> u32 {
                        _rt::Resource::take_handle(&self.handle)
                    }
                    #[doc(hidden)]
                    pub fn handle(&self) -> u32 {
                        _rt::Resource::handle(&self.handle)
                    }
                    #[doc(hidden)]
                    fn type_guard<T: 'static>() {
                        use core::any::TypeId;
                        static mut LAST_TYPE: Option<TypeId> = None;
                        unsafe {
                            assert!(! cfg!(target_feature = "atomics"));
                            let id = TypeId::of::<T>();
                            match LAST_TYPE {
                                Some(ty) => {
                                    assert!(
                                        ty == id, "cannot use two types with this resource type"
                                    )
                                }
                                None => LAST_TYPE = Some(id),
                            }
                        }
                    }
                    #[doc(hidden)]
                    pub unsafe fn dtor<T: 'static>(handle: *mut u8) {
                        Self::type_guard::<T>();
                        let _ = _rt::Box::from_raw(handle as *mut _DtwRep<T>);
                    }
                    fn as_ptr<T: GuestDtw>(&self) -> *mut _DtwRep<T> {
                        Dtw::type_guard::<T>();
                        T::_resource_rep(self.handle()).cast()
                    }
                }
                /// A borrowed version of [`Dtw`] which represents a borrowed value
                /// with the lifetime `'a`.
                #[derive(Debug)]
                #[repr(transparent)]
                pub struct DtwBorrow<'a> {
                    rep: *mut u8,
                    _marker: core::marker::PhantomData<&'a Dtw>,
                }
                impl<'a> DtwBorrow<'a> {
                    #[doc(hidden)]
                    pub unsafe fn lift(rep: usize) -> Self {
                        Self {
                            rep: rep as *mut u8,
                            _marker: core::marker::PhantomData,
                        }
                    }
                    /// Gets access to the underlying `T` in this resource.
                    pub fn get<T: GuestDtw>(&self) -> &T {
                        let ptr = unsafe { &mut *self.as_ptr::<T>() };
                        ptr.as_ref().unwrap()
                    }
                    fn as_ptr<T: 'static>(&self) -> *mut _DtwRep<T> {
                        Dtw::type_guard::<T>();
                        self.rep.cast()
                    }
                }
                unsafe impl _rt::WasmResource for Dtw {
                    #[inline]
                    unsafe fn drop(_handle: u32) {
                        #[cfg(not(target_arch = "wasm32"))]
                        unreachable!();
                        #[cfg(target_arch = "wasm32")]
                        {
                            #[link(wasm_import_module = "[export]augurs:dtw/dtw")]
                            extern "C" {
                                #[link_name = "[resource-drop]dtw"]
                                fn drop(_: u32);
                            }
                            drop(_handle);
                        }
                    }
                }
                #[doc(hidden)]
                #[allow(non_snake_case)]
                pub unsafe fn _export_static_dtw_euclidean_cabi<T: GuestDtw>(
                    arg0: i32,
                    arg1: i32,
                    arg2: i32,
                    arg3: i32,
                    arg4: f64,
                    arg5: i32,
                    arg6: f64,
                    arg7: i32,
                    arg8: f64,
                ) -> i32 {
                    #[cfg(target_arch = "wasm32")] _rt::run_ctors_once();
                    let result0 = T::euclidean(
                        match arg0 {
                            0 => None,
                            1 => {
                                let e = DtwOpts {
                                    window: match arg1 {
                                        0 => None,
                                        1 => {
                                            let e = arg2 as u32;
                                            Some(e)
                                        }
                                        _ => _rt::invalid_enum_discriminant(),
                                    },
                                    max_distance: match arg3 {
                                        0 => None,
                                        1 => {
                                            let e = arg4;
                                            Some(e)
                                        }
                                        _ => _rt::invalid_enum_discriminant(),
                                    },
                                    lower_bound: match arg5 {
                                        0 => None,
                                        1 => {
                                            let e = arg6;
                                            Some(e)
                                        }
                                        _ => _rt::invalid_enum_discriminant(),
                                    },
                                    upper_bound: match arg7 {
                                        0 => None,
                                        1 => {
                                            let e = arg8;
                                            Some(e)
                                        }
                                        _ => _rt::invalid_enum_discriminant(),
                                    },
                                };
                                Some(e)
                            }
                            _ => _rt::invalid_enum_discriminant(),
                        },
                    );
                    (result0).take_handle() as i32
                }
                #[doc(hidden)]
                #[allow(non_snake_case)]
                pub unsafe fn _export_static_dtw_manhattan_cabi<T: GuestDtw>(
                    arg0: i32,
                    arg1: i32,
                    arg2: i32,
                    arg3: i32,
                    arg4: f64,
                    arg5: i32,
                    arg6: f64,
                    arg7: i32,
                    arg8: f64,
                ) -> i32 {
                    #[cfg(target_arch = "wasm32")] _rt::run_ctors_once();
                    let result0 = T::manhattan(
                        match arg0 {
                            0 => None,
                            1 => {
                                let e = DtwOpts {
                                    window: match arg1 {
                                        0 => None,
                                        1 => {
                                            let e = arg2 as u32;
                                            Some(e)
                                        }
                                        _ => _rt::invalid_enum_discriminant(),
                                    },
                                    max_distance: match arg3 {
                                        0 => None,
                                        1 => {
                                            let e = arg4;
                                            Some(e)
                                        }
                                        _ => _rt::invalid_enum_discriminant(),
                                    },
                                    lower_bound: match arg5 {
                                        0 => None,
                                        1 => {
                                            let e = arg6;
                                            Some(e)
                                        }
                                        _ => _rt::invalid_enum_discriminant(),
                                    },
                                    upper_bound: match arg7 {
                                        0 => None,
                                        1 => {
                                            let e = arg8;
                                            Some(e)
                                        }
                                        _ => _rt::invalid_enum_discriminant(),
                                    },
                                };
                                Some(e)
                            }
                            _ => _rt::invalid_enum_discriminant(),
                        },
                    );
                    (result0).take_handle() as i32
                }
                #[doc(hidden)]
                #[allow(non_snake_case)]
                pub unsafe fn _export_method_dtw_distance_cabi<T: GuestDtw>(
                    arg0: *mut u8,
                    arg1: *mut u8,
                    arg2: usize,
                    arg3: *mut u8,
                    arg4: usize,
                ) -> f64 {
                    #[cfg(target_arch = "wasm32")] _rt::run_ctors_once();
                    let len0 = arg2;
                    let len1 = arg4;
                    let result2 = T::distance(
                        DtwBorrow::lift(arg0 as u32 as usize).get(),
                        _rt::Vec::from_raw_parts(arg1.cast(), len0, len0),
                        _rt::Vec::from_raw_parts(arg3.cast(), len1, len1),
                    );
                    _rt::as_f64(result2)
                }
                #[doc(hidden)]
                #[allow(non_snake_case)]
                pub unsafe fn _export_method_dtw_distance_matrix_cabi<T: GuestDtw>(
                    arg0: *mut u8,
                    arg1: *mut u8,
                    arg2: usize,
                ) -> *mut u8 {
                    #[cfg(target_arch = "wasm32")] _rt::run_ctors_once();
                    let base3 = arg1;
                    let len3 = arg2;
                    let mut result3 = _rt::Vec::with_capacity(len3);
                    for i in 0..len3 {
                        let base = base3.add(i * 8);
                        let e3 = {
                            let l0 = *base.add(0).cast::<*mut u8>();
                            let l1 = *base.add(4).cast::<usize>();
                            let len2 = l1;
                            _rt::Vec::from_raw_parts(l0.cast(), len2, len2)
                        };
                        result3.push(e3);
                    }
                    _rt::cabi_dealloc(base3, len3 * 8, 4);
                    let result4 = T::distance_matrix(
                        DtwBorrow::lift(arg0 as u32 as usize).get(),
                        result3,
                    );
                    let ptr5 = _RET_AREA.0.as_mut_ptr().cast::<u8>();
                    let vec7 = result4;
                    let len7 = vec7.len();
                    let layout7 = _rt::alloc::Layout::from_size_align_unchecked(
                        vec7.len() * 8,
                        4,
                    );
                    let result7 = if layout7.size() != 0 {
                        let ptr = _rt::alloc::alloc(layout7).cast::<u8>();
                        if ptr.is_null() {
                            _rt::alloc::handle_alloc_error(layout7);
                        }
                        ptr
                    } else {
                        ::core::ptr::null_mut()
                    };
                    for (i, e) in vec7.into_iter().enumerate() {
                        let base = result7.add(i * 8);
                        {
                            let vec6 = (e).into_boxed_slice();
                            let ptr6 = vec6.as_ptr().cast::<u8>();
                            let len6 = vec6.len();
                            ::core::mem::forget(vec6);
                            *base.add(4).cast::<usize>() = len6;
                            *base.add(0).cast::<*mut u8>() = ptr6.cast_mut();
                        }
                    }
                    *ptr5.add(4).cast::<usize>() = len7;
                    *ptr5.add(0).cast::<*mut u8>() = result7;
                    ptr5
                }
                #[doc(hidden)]
                #[allow(non_snake_case)]
                pub unsafe fn __post_return_method_dtw_distance_matrix<T: GuestDtw>(
                    arg0: *mut u8,
                ) {
                    let l0 = *arg0.add(0).cast::<*mut u8>();
                    let l1 = *arg0.add(4).cast::<usize>();
                    let base5 = l0;
                    let len5 = l1;
                    for i in 0..len5 {
                        let base = base5.add(i * 8);
                        {
                            let l2 = *base.add(0).cast::<*mut u8>();
                            let l3 = *base.add(4).cast::<usize>();
                            let base4 = l2;
                            let len4 = l3;
                            _rt::cabi_dealloc(base4, len4 * 8, 8);
                        }
                    }
                    _rt::cabi_dealloc(base5, len5 * 8, 4);
                }
                pub trait Guest {
                    type Dtw: GuestDtw;
                }
                pub trait GuestDtw: 'static {
                    #[doc(hidden)]
                    unsafe fn _resource_new(val: *mut u8) -> u32
                    where
                        Self: Sized,
                    {
                        #[cfg(not(target_arch = "wasm32"))]
                        {
                            let _ = val;
                            unreachable!();
                        }
                        #[cfg(target_arch = "wasm32")]
                        {
                            #[link(wasm_import_module = "[export]augurs:dtw/dtw")]
                            extern "C" {
                                #[link_name = "[resource-new]dtw"]
                                fn new(_: *mut u8) -> u32;
                            }
                            new(val)
                        }
                    }
                    #[doc(hidden)]
                    fn _resource_rep(handle: u32) -> *mut u8
                    where
                        Self: Sized,
                    {
                        #[cfg(not(target_arch = "wasm32"))]
                        {
                            let _ = handle;
                            unreachable!();
                        }
                        #[cfg(target_arch = "wasm32")]
                        {
                            #[link(wasm_import_module = "[export]augurs:dtw/dtw")]
                            extern "C" {
                                #[link_name = "[resource-rep]dtw"]
                                fn rep(_: u32) -> *mut u8;
                            }
                            unsafe { rep(handle) }
                        }
                    }
                    /// Create a new DTW calculator using the Euclidean distance.
                    fn euclidean(opts: Option<DtwOpts>) -> Dtw;
                    /// Create a new DTW calculator using the Manhattan distance.
                    fn manhattan(opts: Option<DtwOpts>) -> Dtw;
                    /// Compute the distance between two sequences under Dynamic Time Warping.
                    fn distance(&self, s: TimeSeries, t: TimeSeries) -> f64;
                    /// Compute the distance matrix between all pairs of series.
                    ///
                    /// The series do not all have to be the same length.
                    fn distance_matrix(
                        &self,
                        series: _rt::Vec<TimeSeries>,
                    ) -> DistanceMatrix;
                }
                #[doc(hidden)]
                macro_rules! __export_augurs_dtw_dtw_cabi {
                    ($ty:ident with_types_in $($path_to_types:tt)*) => {
                        const _ : () = { #[export_name =
                        "augurs:dtw/dtw#[static]dtw.euclidean"] unsafe extern "C" fn
                        export_static_dtw_euclidean(arg0 : i32, arg1 : i32, arg2 : i32,
                        arg3 : i32, arg4 : f64, arg5 : i32, arg6 : f64, arg7 : i32, arg8
                        : f64,) -> i32 { $($path_to_types)*::
                        _export_static_dtw_euclidean_cabi::<<$ty as $($path_to_types)*::
                        Guest >::Dtw > (arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7,
                        arg8) } #[export_name = "augurs:dtw/dtw#[static]dtw.manhattan"]
                        unsafe extern "C" fn export_static_dtw_manhattan(arg0 : i32, arg1
                        : i32, arg2 : i32, arg3 : i32, arg4 : f64, arg5 : i32, arg6 :
                        f64, arg7 : i32, arg8 : f64,) -> i32 { $($path_to_types)*::
                        _export_static_dtw_manhattan_cabi::<<$ty as $($path_to_types)*::
                        Guest >::Dtw > (arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7,
                        arg8) } #[export_name = "augurs:dtw/dtw#[method]dtw.distance"]
                        unsafe extern "C" fn export_method_dtw_distance(arg0 : * mut u8,
                        arg1 : * mut u8, arg2 : usize, arg3 : * mut u8, arg4 : usize,) ->
                        f64 { $($path_to_types)*::
                        _export_method_dtw_distance_cabi::<<$ty as $($path_to_types)*::
                        Guest >::Dtw > (arg0, arg1, arg2, arg3, arg4) } #[export_name =
                        "augurs:dtw/dtw#[method]dtw.distance-matrix"] unsafe extern "C"
                        fn export_method_dtw_distance_matrix(arg0 : * mut u8, arg1 : *
                        mut u8, arg2 : usize,) -> * mut u8 { $($path_to_types)*::
                        _export_method_dtw_distance_matrix_cabi::<<$ty as
                        $($path_to_types)*:: Guest >::Dtw > (arg0, arg1, arg2) }
                        #[export_name =
                        "cabi_post_augurs:dtw/dtw#[method]dtw.distance-matrix"] unsafe
                        extern "C" fn _post_return_method_dtw_distance_matrix(arg0 : *
                        mut u8,) { $($path_to_types)*::
                        __post_return_method_dtw_distance_matrix::<<$ty as
                        $($path_to_types)*:: Guest >::Dtw > (arg0) } const _ : () = {
                        #[doc(hidden)] #[export_name = "augurs:dtw/dtw#[dtor]dtw"]
                        #[allow(non_snake_case)] unsafe extern "C" fn dtor(rep : * mut
                        u8) { $($path_to_types)*:: Dtw::dtor::< <$ty as
                        $($path_to_types)*:: Guest >::Dtw > (rep) } }; };
                    };
                }
                #[doc(hidden)]
                pub(crate) use __export_augurs_dtw_dtw_cabi;
                #[repr(align(4))]
                struct _RetArea([::core::mem::MaybeUninit<u8>; 8]);
                static mut _RET_AREA: _RetArea = _RetArea(
                    [::core::mem::MaybeUninit::uninit(); 8],
                );
            }
        }
    }
}
mod _rt {
    pub use alloc_crate::vec::Vec;
    use core::fmt;
    use core::marker;
    use core::sync::atomic::{AtomicU32, Ordering::Relaxed};
    /// A type which represents a component model resource, either imported or
    /// exported into this component.
    ///
    /// This is a low-level wrapper which handles the lifetime of the resource
    /// (namely this has a destructor). The `T` provided defines the component model
    /// intrinsics that this wrapper uses.
    ///
    /// One of the chief purposes of this type is to provide `Deref` implementations
    /// to access the underlying data when it is owned.
    ///
    /// This type is primarily used in generated code for exported and imported
    /// resources.
    #[repr(transparent)]
    pub struct Resource<T: WasmResource> {
        handle: AtomicU32,
        _marker: marker::PhantomData<T>,
    }
    /// A trait which all wasm resources implement, namely providing the ability to
    /// drop a resource.
    ///
    /// This generally is implemented by generated code, not user-facing code.
    #[allow(clippy::missing_safety_doc)]
    pub unsafe trait WasmResource {
        /// Invokes the `[resource-drop]...` intrinsic.
        unsafe fn drop(handle: u32);
    }
    impl<T: WasmResource> Resource<T> {
        #[doc(hidden)]
        pub unsafe fn from_handle(handle: u32) -> Self {
            debug_assert!(handle != u32::MAX);
            Self {
                handle: AtomicU32::new(handle),
                _marker: marker::PhantomData,
            }
        }
        /// Takes ownership of the handle owned by `resource`.
        ///
        /// Note that this ideally would be `into_handle` taking `Resource<T>` by
        /// ownership. The code generator does not enable that in all situations,
        /// unfortunately, so this is provided instead.
        ///
        /// Also note that `take_handle` is in theory only ever called on values
        /// owned by a generated function. For example a generated function might
        /// take `Resource<T>` as an argument but then call `take_handle` on a
        /// reference to that argument. In that sense the dynamic nature of
        /// `take_handle` should only be exposed internally to generated code, not
        /// to user code.
        #[doc(hidden)]
        pub fn take_handle(resource: &Resource<T>) -> u32 {
            resource.handle.swap(u32::MAX, Relaxed)
        }
        #[doc(hidden)]
        pub fn handle(resource: &Resource<T>) -> u32 {
            resource.handle.load(Relaxed)
        }
    }
    impl<T: WasmResource> fmt::Debug for Resource<T> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("Resource").field("handle", &self.handle).finish()
        }
    }
    impl<T: WasmResource> Drop for Resource<T> {
        fn drop(&mut self) {
            unsafe {
                match self.handle.load(Relaxed) {
                    u32::MAX => {}
                    other => T::drop(other),
                }
            }
        }
    }
    pub use alloc_crate::boxed::Box;
    #[cfg(target_arch = "wasm32")]
    pub fn run_ctors_once() {
        wit_bindgen_rt::run_ctors_once();
    }
    pub unsafe fn invalid_enum_discriminant<T>() -> T {
        if cfg!(debug_assertions) {
            panic!("invalid enum discriminant")
        } else {
            core::hint::unreachable_unchecked()
        }
    }
    pub fn as_f64<T: AsF64>(t: T) -> f64 {
        t.as_f64()
    }
    pub trait AsF64 {
        fn as_f64(self) -> f64;
    }
    impl<'a, T: Copy + AsF64> AsF64 for &'a T {
        fn as_f64(self) -> f64 {
            (*self).as_f64()
        }
    }
    impl AsF64 for f64 {
        #[inline]
        fn as_f64(self) -> f64 {
            self as f64
        }
    }
    pub unsafe fn cabi_dealloc(ptr: *mut u8, size: usize, align: usize) {
        if size == 0 {
            return;
        }
        let layout = alloc::Layout::from_size_align_unchecked(size, align);
        alloc::dealloc(ptr, layout);
    }
    pub use alloc_crate::alloc;
    extern crate alloc as alloc_crate;
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
        exports::augurs::dtw::dtw::__export_augurs_dtw_dtw_cabi!($ty with_types_in
        $($path_to_types_root)*:: exports::augurs::dtw::dtw);
    };
}
#[doc(inline)]
pub(crate) use __export_component_impl as export;
#[cfg(target_arch = "wasm32")]
#[link_section = "component-type:wit-bindgen:0.35.0:augurs:dtw:component:encoded world"]
#[doc(hidden)]
pub static __WIT_BINDGEN_COMPONENT_TYPE: [u8; 624] = *b"\
\0asm\x0d\0\x01\0\0\x19\x16wit-component-encoding\x04\0\x07\xf0\x03\x01A\x02\x01\
A\x06\x01B\x07\x01pu\x04\0\x0btime-series\x03\0\0\x01pu\x01p\x02\x04\0\x0fdistan\
ce-matrix\x03\0\x03\x01q\x01\x17invalid-distance-matrix\0\0\x04\0\x05error\x03\0\
\x05\x03\0\x11augurs:core/types\x05\0\x02\x03\0\0\x0btime-series\x02\x03\0\0\x0f\
distance-matrix\x01B\x14\x02\x03\x02\x01\x01\x04\0\x0btime-series\x03\0\0\x02\x03\
\x02\x01\x02\x04\0\x0fdistance-matrix\x03\0\x02\x01ky\x01ku\x01r\x04\x06window\x04\
\x0cmax-distance\x05\x0blower-bound\x05\x0bupper-bound\x05\x04\0\x08dtw-opts\x03\
\0\x06\x04\0\x03dtw\x03\x01\x01k\x07\x01i\x08\x01@\x01\x04opts\x09\0\x0a\x04\0\x15\
[static]dtw.euclidean\x01\x0b\x04\0\x15[static]dtw.manhattan\x01\x0b\x01h\x08\x01\
@\x03\x04self\x0c\x01s\x01\x01t\x01\0u\x04\0\x14[method]dtw.distance\x01\x0d\x01\
p\x01\x01@\x02\x04self\x0c\x06series\x0e\0\x03\x04\0\x1b[method]dtw.distance-mat\
rix\x01\x0f\x04\0\x0eaugurs:dtw/dtw\x05\x03\x04\0\x14augurs:dtw/component\x04\0\x0b\
\x0f\x01\0\x09component\x03\0\0\0G\x09producers\x01\x0cprocessed-by\x02\x0dwit-c\
omponent\x070.220.0\x10wit-bindgen-rust\x060.35.0";
#[inline(never)]
#[doc(hidden)]
pub fn __link_custom_section_describing_imports() {
    wit_bindgen_rt::maybe_link_cabi_realloc();
}
