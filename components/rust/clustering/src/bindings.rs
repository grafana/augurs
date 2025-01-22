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
            /// A matrix of pairwise distances between time series.
            pub type DistanceMatrix = _rt::Vec<_rt::Vec<f64>>;
            #[derive(Clone, Copy)]
            pub enum Error {
                /// The distance matrix is not square.
                InvalidDistanceMatrix,
            }
            impl ::core::fmt::Debug for Error {
                fn fmt(
                    &self,
                    f: &mut ::core::fmt::Formatter<'_>,
                ) -> ::core::fmt::Result {
                    match self {
                        Error::InvalidDistanceMatrix => {
                            f.debug_tuple("Error::InvalidDistanceMatrix").finish()
                        }
                    }
                }
            }
            impl ::core::fmt::Display for Error {
                fn fmt(
                    &self,
                    f: &mut ::core::fmt::Formatter<'_>,
                ) -> ::core::fmt::Result {
                    write!(f, "{:?}", self)
                }
            }
            impl std::error::Error for Error {}
        }
    }
}
#[allow(dead_code)]
pub mod exports {
    #[allow(dead_code)]
    pub mod augurs {
        #[allow(dead_code)]
        pub mod clustering {
            #[allow(dead_code, clippy::all)]
            pub mod dbscan {
                #[used]
                #[doc(hidden)]
                static __FORCE_SECTION_REF: fn() = super::super::super::super::__link_custom_section_describing_imports;
                use super::super::super::super::_rt;
                pub type DistanceMatrix = super::super::super::super::augurs::core::types::DistanceMatrix;
                pub type Error = super::super::super::super::augurs::core::types::Error;
                /// Options for DBSCAN clustering.
                #[repr(C)]
                #[derive(Clone, Copy)]
                pub struct Options {
                    /// The maximum distance between two series for them to be considered as in the same cluster.
                    pub epsilon: f64,
                    /// The minimum number of series before a cluster is considered core.
                    pub minimum_cluster_size: u32,
                }
                impl ::core::fmt::Debug for Options {
                    fn fmt(
                        &self,
                        f: &mut ::core::fmt::Formatter<'_>,
                    ) -> ::core::fmt::Result {
                        f.debug_struct("Options")
                            .field("epsilon", &self.epsilon)
                            .field("minimum-cluster-size", &self.minimum_cluster_size)
                            .finish()
                    }
                }
                #[derive(Clone, Copy)]
                pub enum DbscanCluster {
                    Noise,
                    Cluster(u32),
                }
                impl ::core::fmt::Debug for DbscanCluster {
                    fn fmt(
                        &self,
                        f: &mut ::core::fmt::Formatter<'_>,
                    ) -> ::core::fmt::Result {
                        match self {
                            DbscanCluster::Noise => {
                                f.debug_tuple("DbscanCluster::Noise").finish()
                            }
                            DbscanCluster::Cluster(e) => {
                                f.debug_tuple("DbscanCluster::Cluster").field(e).finish()
                            }
                        }
                    }
                }
                /// A DBSCAN clusterer.
                #[derive(Debug)]
                #[repr(transparent)]
                pub struct Dbscan {
                    handle: _rt::Resource<Dbscan>,
                }
                type _DbscanRep<T> = Option<T>;
                impl Dbscan {
                    /// Creates a new resource from the specified representation.
                    ///
                    /// This function will create a new resource handle by moving `val` onto
                    /// the heap and then passing that heap pointer to the component model to
                    /// create a handle. The owned handle is then returned as `Dbscan`.
                    pub fn new<T: GuestDbscan>(val: T) -> Self {
                        Self::type_guard::<T>();
                        let val: _DbscanRep<T> = Some(val);
                        let ptr: *mut _DbscanRep<T> = _rt::Box::into_raw(
                            _rt::Box::new(val),
                        );
                        unsafe { Self::from_handle(T::_resource_new(ptr.cast())) }
                    }
                    /// Gets access to the underlying `T` which represents this resource.
                    pub fn get<T: GuestDbscan>(&self) -> &T {
                        let ptr = unsafe { &*self.as_ptr::<T>() };
                        ptr.as_ref().unwrap()
                    }
                    /// Gets mutable access to the underlying `T` which represents this
                    /// resource.
                    pub fn get_mut<T: GuestDbscan>(&mut self) -> &mut T {
                        let ptr = unsafe { &mut *self.as_ptr::<T>() };
                        ptr.as_mut().unwrap()
                    }
                    /// Consumes this resource and returns the underlying `T`.
                    pub fn into_inner<T: GuestDbscan>(self) -> T {
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
                        let _ = _rt::Box::from_raw(handle as *mut _DbscanRep<T>);
                    }
                    fn as_ptr<T: GuestDbscan>(&self) -> *mut _DbscanRep<T> {
                        Dbscan::type_guard::<T>();
                        T::_resource_rep(self.handle()).cast()
                    }
                }
                /// A borrowed version of [`Dbscan`] which represents a borrowed value
                /// with the lifetime `'a`.
                #[derive(Debug)]
                #[repr(transparent)]
                pub struct DbscanBorrow<'a> {
                    rep: *mut u8,
                    _marker: core::marker::PhantomData<&'a Dbscan>,
                }
                impl<'a> DbscanBorrow<'a> {
                    #[doc(hidden)]
                    pub unsafe fn lift(rep: usize) -> Self {
                        Self {
                            rep: rep as *mut u8,
                            _marker: core::marker::PhantomData,
                        }
                    }
                    /// Gets access to the underlying `T` in this resource.
                    pub fn get<T: GuestDbscan>(&self) -> &T {
                        let ptr = unsafe { &mut *self.as_ptr::<T>() };
                        ptr.as_ref().unwrap()
                    }
                    fn as_ptr<T: 'static>(&self) -> *mut _DbscanRep<T> {
                        Dbscan::type_guard::<T>();
                        self.rep.cast()
                    }
                }
                unsafe impl _rt::WasmResource for Dbscan {
                    #[inline]
                    unsafe fn drop(_handle: u32) {
                        #[cfg(not(target_arch = "wasm32"))]
                        unreachable!();
                        #[cfg(target_arch = "wasm32")]
                        {
                            #[link(
                                wasm_import_module = "[export]augurs:clustering/dbscan"
                            )]
                            extern "C" {
                                #[link_name = "[resource-drop]dbscan"]
                                fn drop(_: u32);
                            }
                            drop(_handle);
                        }
                    }
                }
                #[doc(hidden)]
                #[allow(non_snake_case)]
                pub unsafe fn _export_constructor_dbscan_cabi<T: GuestDbscan>(
                    arg0: f64,
                    arg1: i32,
                ) -> i32 {
                    #[cfg(target_arch = "wasm32")] _rt::run_ctors_once();
                    let result0 = Dbscan::new(
                        T::new(Options {
                            epsilon: arg0,
                            minimum_cluster_size: arg1 as u32,
                        }),
                    );
                    (result0).take_handle() as i32
                }
                #[doc(hidden)]
                #[allow(non_snake_case)]
                pub unsafe fn _export_method_dbscan_fit_cabi<T: GuestDbscan>(
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
                    let result4 = T::fit(
                        DbscanBorrow::lift(arg0 as u32 as usize).get(),
                        result3,
                    );
                    let ptr5 = _RET_AREA.0.as_mut_ptr().cast::<u8>();
                    match result4 {
                        Ok(e) => {
                            *ptr5.add(0).cast::<u8>() = (0i32) as u8;
                            let vec6 = e;
                            let len6 = vec6.len();
                            let layout6 = _rt::alloc::Layout::from_size_align_unchecked(
                                vec6.len() * 8,
                                4,
                            );
                            let result6 = if layout6.size() != 0 {
                                let ptr = _rt::alloc::alloc(layout6).cast::<u8>();
                                if ptr.is_null() {
                                    _rt::alloc::handle_alloc_error(layout6);
                                }
                                ptr
                            } else {
                                ::core::ptr::null_mut()
                            };
                            for (i, e) in vec6.into_iter().enumerate() {
                                let base = result6.add(i * 8);
                                {
                                    match e {
                                        DbscanCluster::Noise => {
                                            *base.add(0).cast::<u8>() = (0i32) as u8;
                                        }
                                        DbscanCluster::Cluster(e) => {
                                            *base.add(0).cast::<u8>() = (1i32) as u8;
                                            *base.add(4).cast::<i32>() = _rt::as_i32(e);
                                        }
                                    }
                                }
                            }
                            *ptr5.add(8).cast::<usize>() = len6;
                            *ptr5.add(4).cast::<*mut u8>() = result6;
                        }
                        Err(e) => {
                            *ptr5.add(0).cast::<u8>() = (1i32) as u8;
                            use super::super::super::super::augurs::core::types::Error as V7;
                            match e {
                                V7::InvalidDistanceMatrix => {
                                    *ptr5.add(4).cast::<u8>() = (0i32) as u8;
                                }
                            }
                        }
                    };
                    ptr5
                }
                #[doc(hidden)]
                #[allow(non_snake_case)]
                pub unsafe fn __post_return_method_dbscan_fit<T: GuestDbscan>(
                    arg0: *mut u8,
                ) {
                    let l0 = i32::from(*arg0.add(0).cast::<u8>());
                    match l0 {
                        0 => {
                            let l1 = *arg0.add(4).cast::<*mut u8>();
                            let l2 = *arg0.add(8).cast::<usize>();
                            let base3 = l1;
                            let len3 = l2;
                            _rt::cabi_dealloc(base3, len3 * 8, 4);
                        }
                        _ => {}
                    }
                }
                pub trait Guest {
                    type Dbscan: GuestDbscan;
                }
                pub trait GuestDbscan: 'static {
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
                            #[link(
                                wasm_import_module = "[export]augurs:clustering/dbscan"
                            )]
                            extern "C" {
                                #[link_name = "[resource-new]dbscan"]
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
                            #[link(
                                wasm_import_module = "[export]augurs:clustering/dbscan"
                            )]
                            extern "C" {
                                #[link_name = "[resource-rep]dbscan"]
                                fn rep(_: u32) -> *mut u8;
                            }
                            unsafe { rep(handle) }
                        }
                    }
                    /// Create a new DBSCAN clusterer.
                    fn new(options: Options) -> Self;
                    fn fit(
                        &self,
                        matrix: DistanceMatrix,
                    ) -> Result<_rt::Vec<DbscanCluster>, Error>;
                }
                #[doc(hidden)]
                macro_rules! __export_augurs_clustering_dbscan_cabi {
                    ($ty:ident with_types_in $($path_to_types:tt)*) => {
                        const _ : () = { #[export_name =
                        "augurs:clustering/dbscan#[constructor]dbscan"] unsafe extern "C"
                        fn export_constructor_dbscan(arg0 : f64, arg1 : i32,) -> i32 {
                        $($path_to_types)*:: _export_constructor_dbscan_cabi::<<$ty as
                        $($path_to_types)*:: Guest >::Dbscan > (arg0, arg1) }
                        #[export_name = "augurs:clustering/dbscan#[method]dbscan.fit"]
                        unsafe extern "C" fn export_method_dbscan_fit(arg0 : * mut u8,
                        arg1 : * mut u8, arg2 : usize,) -> * mut u8 {
                        $($path_to_types)*:: _export_method_dbscan_fit_cabi::<<$ty as
                        $($path_to_types)*:: Guest >::Dbscan > (arg0, arg1, arg2) }
                        #[export_name =
                        "cabi_post_augurs:clustering/dbscan#[method]dbscan.fit"] unsafe
                        extern "C" fn _post_return_method_dbscan_fit(arg0 : * mut u8,) {
                        $($path_to_types)*:: __post_return_method_dbscan_fit::<<$ty as
                        $($path_to_types)*:: Guest >::Dbscan > (arg0) } const _ : () = {
                        #[doc(hidden)] #[export_name =
                        "augurs:clustering/dbscan#[dtor]dbscan"] #[allow(non_snake_case)]
                        unsafe extern "C" fn dtor(rep : * mut u8) { $($path_to_types)*::
                        Dbscan::dtor::< <$ty as $($path_to_types)*:: Guest >::Dbscan >
                        (rep) } }; };
                    };
                }
                #[doc(hidden)]
                pub(crate) use __export_augurs_clustering_dbscan_cabi;
                #[repr(align(4))]
                struct _RetArea([::core::mem::MaybeUninit<u8>; 12]);
                static mut _RET_AREA: _RetArea = _RetArea(
                    [::core::mem::MaybeUninit::uninit(); 12],
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
    pub unsafe fn cabi_dealloc(ptr: *mut u8, size: usize, align: usize) {
        if size == 0 {
            return;
        }
        let layout = alloc::Layout::from_size_align_unchecked(size, align);
        alloc::dealloc(ptr, layout);
    }
    pub fn as_i32<T: AsI32>(t: T) -> i32 {
        t.as_i32()
    }
    pub trait AsI32 {
        fn as_i32(self) -> i32;
    }
    impl<'a, T: Copy + AsI32> AsI32 for &'a T {
        fn as_i32(self) -> i32 {
            (*self).as_i32()
        }
    }
    impl AsI32 for i32 {
        #[inline]
        fn as_i32(self) -> i32 {
            self as i32
        }
    }
    impl AsI32 for u32 {
        #[inline]
        fn as_i32(self) -> i32 {
            self as i32
        }
    }
    impl AsI32 for i16 {
        #[inline]
        fn as_i32(self) -> i32 {
            self as i32
        }
    }
    impl AsI32 for u16 {
        #[inline]
        fn as_i32(self) -> i32 {
            self as i32
        }
    }
    impl AsI32 for i8 {
        #[inline]
        fn as_i32(self) -> i32 {
            self as i32
        }
    }
    impl AsI32 for u8 {
        #[inline]
        fn as_i32(self) -> i32 {
            self as i32
        }
    }
    impl AsI32 for char {
        #[inline]
        fn as_i32(self) -> i32 {
            self as i32
        }
    }
    impl AsI32 for usize {
        #[inline]
        fn as_i32(self) -> i32 {
            self as i32
        }
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
        exports::augurs::clustering::dbscan::__export_augurs_clustering_dbscan_cabi!($ty
        with_types_in $($path_to_types_root)*:: exports::augurs::clustering::dbscan);
    };
}
#[doc(inline)]
pub(crate) use __export_component_impl as export;
#[cfg(target_arch = "wasm32")]
#[link_section = "component-type:wit-bindgen:0.35.0:augurs:clustering:component:encoded world"]
#[doc(hidden)]
pub static __WIT_BINDGEN_COMPONENT_TYPE: [u8; 577] = *b"\
\0asm\x0d\0\x01\0\0\x19\x16wit-component-encoding\x04\0\x07\xc1\x03\x01A\x02\x01\
A\x06\x01B\x07\x01pu\x04\0\x0btime-series\x03\0\0\x01pu\x01p\x02\x04\0\x0fdistan\
ce-matrix\x03\0\x03\x01q\x01\x17invalid-distance-matrix\0\0\x04\0\x05error\x03\0\
\x05\x03\0\x11augurs:core/types\x05\0\x02\x03\0\0\x0fdistance-matrix\x02\x03\0\0\
\x05error\x01B\x11\x02\x03\x02\x01\x01\x04\0\x0fdistance-matrix\x03\0\0\x02\x03\x02\
\x01\x02\x04\0\x05error\x03\0\x02\x01r\x02\x07epsilonu\x14minimum-cluster-sizey\x04\
\0\x07options\x03\0\x04\x01q\x02\x05noise\0\0\x07cluster\x01y\0\x04\0\x0edbscan-\
cluster\x03\0\x06\x04\0\x06dbscan\x03\x01\x01i\x08\x01@\x01\x07options\x05\0\x09\
\x04\0\x13[constructor]dbscan\x01\x0a\x01h\x08\x01p\x07\x01j\x01\x0c\x01\x03\x01\
@\x02\x04self\x0b\x06matrix\x01\0\x0d\x04\0\x12[method]dbscan.fit\x01\x0e\x04\0\x18\
augurs:clustering/dbscan\x05\x03\x04\0\x1baugurs:clustering/component\x04\0\x0b\x0f\
\x01\0\x09component\x03\0\0\0G\x09producers\x01\x0cprocessed-by\x02\x0dwit-compo\
nent\x070.220.0\x10wit-bindgen-rust\x060.35.0";
#[inline(never)]
#[doc(hidden)]
pub fn __link_custom_section_describing_imports() {
    wit_bindgen_rt::maybe_link_cabi_realloc();
}
