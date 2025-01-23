#[allow(dead_code)]
pub mod exports {
    #[allow(dead_code)]
    pub mod augurs {
        #[allow(dead_code)]
        pub mod core {
            #[allow(dead_code, clippy::all)]
            pub mod types {
                #[used]
                #[doc(hidden)]
                static __FORCE_SECTION_REF: fn() = super::super::super::super::__link_custom_section_describing_imports;
                use super::super::super::super::_rt;
                /// A sequence of floats comprising a time series.
                pub type TimeSeries = _rt::Vec<f64>;
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
                #[derive(Clone)]
                pub struct ForecastIntervals {
                    pub level: f64,
                    pub lower: TimeSeries,
                    pub upper: TimeSeries,
                }
                impl ::core::fmt::Debug for ForecastIntervals {
                    fn fmt(
                        &self,
                        f: &mut ::core::fmt::Formatter<'_>,
                    ) -> ::core::fmt::Result {
                        f.debug_struct("ForecastIntervals")
                            .field("level", &self.level)
                            .field("lower", &self.lower)
                            .field("upper", &self.upper)
                            .finish()
                    }
                }
                #[derive(Clone)]
                pub struct Forecast {
                    pub point: TimeSeries,
                    pub intervals: Option<ForecastIntervals>,
                }
                impl ::core::fmt::Debug for Forecast {
                    fn fmt(
                        &self,
                        f: &mut ::core::fmt::Formatter<'_>,
                    ) -> ::core::fmt::Result {
                        f.debug_struct("Forecast")
                            .field("point", &self.point)
                            .field("intervals", &self.intervals)
                            .finish()
                    }
                }
                #[derive(Debug)]
                #[repr(transparent)]
                pub struct FittedTrendModel {
                    handle: _rt::Resource<FittedTrendModel>,
                }
                type _FittedTrendModelRep<T> = Option<T>;
                impl FittedTrendModel {
                    /// Creates a new resource from the specified representation.
                    ///
                    /// This function will create a new resource handle by moving `val` onto
                    /// the heap and then passing that heap pointer to the component model to
                    /// create a handle. The owned handle is then returned as `FittedTrendModel`.
                    pub fn new<T: GuestFittedTrendModel>(val: T) -> Self {
                        Self::type_guard::<T>();
                        let val: _FittedTrendModelRep<T> = Some(val);
                        let ptr: *mut _FittedTrendModelRep<T> = _rt::Box::into_raw(
                            _rt::Box::new(val),
                        );
                        unsafe { Self::from_handle(T::_resource_new(ptr.cast())) }
                    }
                    /// Gets access to the underlying `T` which represents this resource.
                    pub fn get<T: GuestFittedTrendModel>(&self) -> &T {
                        let ptr = unsafe { &*self.as_ptr::<T>() };
                        ptr.as_ref().unwrap()
                    }
                    /// Gets mutable access to the underlying `T` which represents this
                    /// resource.
                    pub fn get_mut<T: GuestFittedTrendModel>(&mut self) -> &mut T {
                        let ptr = unsafe { &mut *self.as_ptr::<T>() };
                        ptr.as_mut().unwrap()
                    }
                    /// Consumes this resource and returns the underlying `T`.
                    pub fn into_inner<T: GuestFittedTrendModel>(self) -> T {
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
                        let _ = _rt::Box::from_raw(
                            handle as *mut _FittedTrendModelRep<T>,
                        );
                    }
                    fn as_ptr<T: GuestFittedTrendModel>(
                        &self,
                    ) -> *mut _FittedTrendModelRep<T> {
                        FittedTrendModel::type_guard::<T>();
                        T::_resource_rep(self.handle()).cast()
                    }
                }
                /// A borrowed version of [`FittedTrendModel`] which represents a borrowed value
                /// with the lifetime `'a`.
                #[derive(Debug)]
                #[repr(transparent)]
                pub struct FittedTrendModelBorrow<'a> {
                    rep: *mut u8,
                    _marker: core::marker::PhantomData<&'a FittedTrendModel>,
                }
                impl<'a> FittedTrendModelBorrow<'a> {
                    #[doc(hidden)]
                    pub unsafe fn lift(rep: usize) -> Self {
                        Self {
                            rep: rep as *mut u8,
                            _marker: core::marker::PhantomData,
                        }
                    }
                    /// Gets access to the underlying `T` in this resource.
                    pub fn get<T: GuestFittedTrendModel>(&self) -> &T {
                        let ptr = unsafe { &mut *self.as_ptr::<T>() };
                        ptr.as_ref().unwrap()
                    }
                    fn as_ptr<T: 'static>(&self) -> *mut _FittedTrendModelRep<T> {
                        FittedTrendModel::type_guard::<T>();
                        self.rep.cast()
                    }
                }
                unsafe impl _rt::WasmResource for FittedTrendModel {
                    #[inline]
                    unsafe fn drop(_handle: u32) {
                        #[cfg(not(target_arch = "wasm32"))]
                        unreachable!();
                        #[cfg(target_arch = "wasm32")]
                        {
                            #[link(wasm_import_module = "[export]augurs:core/types")]
                            extern "C" {
                                #[link_name = "[resource-drop]fitted-trend-model"]
                                fn drop(_: u32);
                            }
                            drop(_handle);
                        }
                    }
                }
                #[derive(Debug)]
                #[repr(transparent)]
                pub struct TrendModel {
                    handle: _rt::Resource<TrendModel>,
                }
                type _TrendModelRep<T> = Option<T>;
                impl TrendModel {
                    /// Creates a new resource from the specified representation.
                    ///
                    /// This function will create a new resource handle by moving `val` onto
                    /// the heap and then passing that heap pointer to the component model to
                    /// create a handle. The owned handle is then returned as `TrendModel`.
                    pub fn new<T: GuestTrendModel>(val: T) -> Self {
                        Self::type_guard::<T>();
                        let val: _TrendModelRep<T> = Some(val);
                        let ptr: *mut _TrendModelRep<T> = _rt::Box::into_raw(
                            _rt::Box::new(val),
                        );
                        unsafe { Self::from_handle(T::_resource_new(ptr.cast())) }
                    }
                    /// Gets access to the underlying `T` which represents this resource.
                    pub fn get<T: GuestTrendModel>(&self) -> &T {
                        let ptr = unsafe { &*self.as_ptr::<T>() };
                        ptr.as_ref().unwrap()
                    }
                    /// Gets mutable access to the underlying `T` which represents this
                    /// resource.
                    pub fn get_mut<T: GuestTrendModel>(&mut self) -> &mut T {
                        let ptr = unsafe { &mut *self.as_ptr::<T>() };
                        ptr.as_mut().unwrap()
                    }
                    /// Consumes this resource and returns the underlying `T`.
                    pub fn into_inner<T: GuestTrendModel>(self) -> T {
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
                        let _ = _rt::Box::from_raw(handle as *mut _TrendModelRep<T>);
                    }
                    fn as_ptr<T: GuestTrendModel>(&self) -> *mut _TrendModelRep<T> {
                        TrendModel::type_guard::<T>();
                        T::_resource_rep(self.handle()).cast()
                    }
                }
                /// A borrowed version of [`TrendModel`] which represents a borrowed value
                /// with the lifetime `'a`.
                #[derive(Debug)]
                #[repr(transparent)]
                pub struct TrendModelBorrow<'a> {
                    rep: *mut u8,
                    _marker: core::marker::PhantomData<&'a TrendModel>,
                }
                impl<'a> TrendModelBorrow<'a> {
                    #[doc(hidden)]
                    pub unsafe fn lift(rep: usize) -> Self {
                        Self {
                            rep: rep as *mut u8,
                            _marker: core::marker::PhantomData,
                        }
                    }
                    /// Gets access to the underlying `T` in this resource.
                    pub fn get<T: GuestTrendModel>(&self) -> &T {
                        let ptr = unsafe { &mut *self.as_ptr::<T>() };
                        ptr.as_ref().unwrap()
                    }
                    fn as_ptr<T: 'static>(&self) -> *mut _TrendModelRep<T> {
                        TrendModel::type_guard::<T>();
                        self.rep.cast()
                    }
                }
                unsafe impl _rt::WasmResource for TrendModel {
                    #[inline]
                    unsafe fn drop(_handle: u32) {
                        #[cfg(not(target_arch = "wasm32"))]
                        unreachable!();
                        #[cfg(target_arch = "wasm32")]
                        {
                            #[link(wasm_import_module = "[export]augurs:core/types")]
                            extern "C" {
                                #[link_name = "[resource-drop]trend-model"]
                                fn drop(_: u32);
                            }
                            drop(_handle);
                        }
                    }
                }
                #[doc(hidden)]
                #[allow(non_snake_case)]
                pub unsafe fn _export_method_fitted_trend_model_predict_in_sample_cabi<
                    T: GuestFittedTrendModel,
                >(arg0: *mut u8, arg1: i32, arg2: f64) -> *mut u8 {
                    #[cfg(target_arch = "wasm32")] _rt::run_ctors_once();
                    let result0 = T::predict_in_sample(
                        FittedTrendModelBorrow::lift(arg0 as u32 as usize).get(),
                        match arg1 {
                            0 => None,
                            1 => {
                                let e = arg2;
                                Some(e)
                            }
                            _ => _rt::invalid_enum_discriminant(),
                        },
                    );
                    let ptr1 = _RET_AREA.0.as_mut_ptr().cast::<u8>();
                    match result0 {
                        Ok(e) => {
                            *ptr1.add(0).cast::<u8>() = (0i32) as u8;
                            let Forecast { point: point2, intervals: intervals2 } = e;
                            let vec3 = (point2).into_boxed_slice();
                            let ptr3 = vec3.as_ptr().cast::<u8>();
                            let len3 = vec3.len();
                            ::core::mem::forget(vec3);
                            *ptr1.add(12).cast::<usize>() = len3;
                            *ptr1.add(8).cast::<*mut u8>() = ptr3.cast_mut();
                            match intervals2 {
                                Some(e) => {
                                    *ptr1.add(16).cast::<u8>() = (1i32) as u8;
                                    let ForecastIntervals {
                                        level: level4,
                                        lower: lower4,
                                        upper: upper4,
                                    } = e;
                                    *ptr1.add(24).cast::<f64>() = _rt::as_f64(level4);
                                    let vec5 = (lower4).into_boxed_slice();
                                    let ptr5 = vec5.as_ptr().cast::<u8>();
                                    let len5 = vec5.len();
                                    ::core::mem::forget(vec5);
                                    *ptr1.add(36).cast::<usize>() = len5;
                                    *ptr1.add(32).cast::<*mut u8>() = ptr5.cast_mut();
                                    let vec6 = (upper4).into_boxed_slice();
                                    let ptr6 = vec6.as_ptr().cast::<u8>();
                                    let len6 = vec6.len();
                                    ::core::mem::forget(vec6);
                                    *ptr1.add(44).cast::<usize>() = len6;
                                    *ptr1.add(40).cast::<*mut u8>() = ptr6.cast_mut();
                                }
                                None => {
                                    *ptr1.add(16).cast::<u8>() = (0i32) as u8;
                                }
                            };
                        }
                        Err(e) => {
                            *ptr1.add(0).cast::<u8>() = (1i32) as u8;
                            match e {
                                Error::InvalidDistanceMatrix => {
                                    *ptr1.add(8).cast::<u8>() = (0i32) as u8;
                                }
                            }
                        }
                    };
                    ptr1
                }
                #[doc(hidden)]
                #[allow(non_snake_case)]
                pub unsafe fn __post_return_method_fitted_trend_model_predict_in_sample<
                    T: GuestFittedTrendModel,
                >(arg0: *mut u8) {
                    let l0 = i32::from(*arg0.add(0).cast::<u8>());
                    match l0 {
                        0 => {
                            let l1 = *arg0.add(8).cast::<*mut u8>();
                            let l2 = *arg0.add(12).cast::<usize>();
                            let base3 = l1;
                            let len3 = l2;
                            _rt::cabi_dealloc(base3, len3 * 8, 8);
                            let l4 = i32::from(*arg0.add(16).cast::<u8>());
                            match l4 {
                                0 => {}
                                _ => {
                                    let l5 = *arg0.add(32).cast::<*mut u8>();
                                    let l6 = *arg0.add(36).cast::<usize>();
                                    let base7 = l5;
                                    let len7 = l6;
                                    _rt::cabi_dealloc(base7, len7 * 8, 8);
                                    let l8 = *arg0.add(40).cast::<*mut u8>();
                                    let l9 = *arg0.add(44).cast::<usize>();
                                    let base10 = l8;
                                    let len10 = l9;
                                    _rt::cabi_dealloc(base10, len10 * 8, 8);
                                }
                            }
                        }
                        _ => {}
                    }
                }
                #[doc(hidden)]
                #[allow(non_snake_case)]
                pub unsafe fn _export_method_fitted_trend_model_predict_cabi<
                    T: GuestFittedTrendModel,
                >(arg0: *mut u8, arg1: i32, arg2: i32, arg3: f64) -> *mut u8 {
                    #[cfg(target_arch = "wasm32")] _rt::run_ctors_once();
                    let result0 = T::predict(
                        FittedTrendModelBorrow::lift(arg0 as u32 as usize).get(),
                        arg1 as u32,
                        match arg2 {
                            0 => None,
                            1 => {
                                let e = arg3;
                                Some(e)
                            }
                            _ => _rt::invalid_enum_discriminant(),
                        },
                    );
                    let ptr1 = _RET_AREA.0.as_mut_ptr().cast::<u8>();
                    match result0 {
                        Ok(e) => {
                            *ptr1.add(0).cast::<u8>() = (0i32) as u8;
                            let Forecast { point: point2, intervals: intervals2 } = e;
                            let vec3 = (point2).into_boxed_slice();
                            let ptr3 = vec3.as_ptr().cast::<u8>();
                            let len3 = vec3.len();
                            ::core::mem::forget(vec3);
                            *ptr1.add(12).cast::<usize>() = len3;
                            *ptr1.add(8).cast::<*mut u8>() = ptr3.cast_mut();
                            match intervals2 {
                                Some(e) => {
                                    *ptr1.add(16).cast::<u8>() = (1i32) as u8;
                                    let ForecastIntervals {
                                        level: level4,
                                        lower: lower4,
                                        upper: upper4,
                                    } = e;
                                    *ptr1.add(24).cast::<f64>() = _rt::as_f64(level4);
                                    let vec5 = (lower4).into_boxed_slice();
                                    let ptr5 = vec5.as_ptr().cast::<u8>();
                                    let len5 = vec5.len();
                                    ::core::mem::forget(vec5);
                                    *ptr1.add(36).cast::<usize>() = len5;
                                    *ptr1.add(32).cast::<*mut u8>() = ptr5.cast_mut();
                                    let vec6 = (upper4).into_boxed_slice();
                                    let ptr6 = vec6.as_ptr().cast::<u8>();
                                    let len6 = vec6.len();
                                    ::core::mem::forget(vec6);
                                    *ptr1.add(44).cast::<usize>() = len6;
                                    *ptr1.add(40).cast::<*mut u8>() = ptr6.cast_mut();
                                }
                                None => {
                                    *ptr1.add(16).cast::<u8>() = (0i32) as u8;
                                }
                            };
                        }
                        Err(e) => {
                            *ptr1.add(0).cast::<u8>() = (1i32) as u8;
                            match e {
                                Error::InvalidDistanceMatrix => {
                                    *ptr1.add(8).cast::<u8>() = (0i32) as u8;
                                }
                            }
                        }
                    };
                    ptr1
                }
                #[doc(hidden)]
                #[allow(non_snake_case)]
                pub unsafe fn __post_return_method_fitted_trend_model_predict<
                    T: GuestFittedTrendModel,
                >(arg0: *mut u8) {
                    let l0 = i32::from(*arg0.add(0).cast::<u8>());
                    match l0 {
                        0 => {
                            let l1 = *arg0.add(8).cast::<*mut u8>();
                            let l2 = *arg0.add(12).cast::<usize>();
                            let base3 = l1;
                            let len3 = l2;
                            _rt::cabi_dealloc(base3, len3 * 8, 8);
                            let l4 = i32::from(*arg0.add(16).cast::<u8>());
                            match l4 {
                                0 => {}
                                _ => {
                                    let l5 = *arg0.add(32).cast::<*mut u8>();
                                    let l6 = *arg0.add(36).cast::<usize>();
                                    let base7 = l5;
                                    let len7 = l6;
                                    _rt::cabi_dealloc(base7, len7 * 8, 8);
                                    let l8 = *arg0.add(40).cast::<*mut u8>();
                                    let l9 = *arg0.add(44).cast::<usize>();
                                    let base10 = l8;
                                    let len10 = l9;
                                    _rt::cabi_dealloc(base10, len10 * 8, 8);
                                }
                            }
                        }
                        _ => {}
                    }
                }
                #[doc(hidden)]
                #[allow(non_snake_case)]
                pub unsafe fn _export_method_trend_model_fit_cabi<T: GuestTrendModel>(
                    arg0: *mut u8,
                    arg1: *mut u8,
                    arg2: usize,
                ) -> *mut u8 {
                    #[cfg(target_arch = "wasm32")] _rt::run_ctors_once();
                    let len0 = arg2;
                    let result1 = T::fit(
                        TrendModelBorrow::lift(arg0 as u32 as usize).get(),
                        _rt::Vec::from_raw_parts(arg1.cast(), len0, len0),
                    );
                    let ptr2 = _RET_AREA.0.as_mut_ptr().cast::<u8>();
                    match result1 {
                        Ok(e) => {
                            *ptr2.add(0).cast::<u8>() = (0i32) as u8;
                            *ptr2.add(4).cast::<i32>() = (e).take_handle() as i32;
                        }
                        Err(e) => {
                            *ptr2.add(0).cast::<u8>() = (1i32) as u8;
                            match e {
                                Error::InvalidDistanceMatrix => {
                                    *ptr2.add(4).cast::<u8>() = (0i32) as u8;
                                }
                            }
                        }
                    };
                    ptr2
                }
                pub trait Guest {
                    type FittedTrendModel: GuestFittedTrendModel;
                    type TrendModel: GuestTrendModel;
                }
                pub trait GuestFittedTrendModel: 'static {
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
                            #[link(wasm_import_module = "[export]augurs:core/types")]
                            extern "C" {
                                #[link_name = "[resource-new]fitted-trend-model"]
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
                            #[link(wasm_import_module = "[export]augurs:core/types")]
                            extern "C" {
                                #[link_name = "[resource-rep]fitted-trend-model"]
                                fn rep(_: u32) -> *mut u8;
                            }
                            unsafe { rep(handle) }
                        }
                    }
                    fn predict_in_sample(
                        &self,
                        level: Option<f64>,
                    ) -> Result<Forecast, Error>;
                    fn predict(
                        &self,
                        horizon: u32,
                        level: Option<f64>,
                    ) -> Result<Forecast, Error>;
                }
                pub trait GuestTrendModel: 'static {
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
                            #[link(wasm_import_module = "[export]augurs:core/types")]
                            extern "C" {
                                #[link_name = "[resource-new]trend-model"]
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
                            #[link(wasm_import_module = "[export]augurs:core/types")]
                            extern "C" {
                                #[link_name = "[resource-rep]trend-model"]
                                fn rep(_: u32) -> *mut u8;
                            }
                            unsafe { rep(handle) }
                        }
                    }
                    fn fit(&self, y: TimeSeries) -> Result<FittedTrendModel, Error>;
                }
                #[doc(hidden)]
                macro_rules! __export_augurs_core_types_cabi {
                    ($ty:ident with_types_in $($path_to_types:tt)*) => {
                        const _ : () = { #[export_name =
                        "augurs:core/types#[method]fitted-trend-model.predict-in-sample"]
                        unsafe extern "C" fn
                        export_method_fitted_trend_model_predict_in_sample(arg0 : * mut
                        u8, arg1 : i32, arg2 : f64,) -> * mut u8 { $($path_to_types)*::
                        _export_method_fitted_trend_model_predict_in_sample_cabi::<<$ty
                        as $($path_to_types)*:: Guest >::FittedTrendModel > (arg0, arg1,
                        arg2) } #[export_name =
                        "cabi_post_augurs:core/types#[method]fitted-trend-model.predict-in-sample"]
                        unsafe extern "C" fn
                        _post_return_method_fitted_trend_model_predict_in_sample(arg0 : *
                        mut u8,) { $($path_to_types)*::
                        __post_return_method_fitted_trend_model_predict_in_sample::<<$ty
                        as $($path_to_types)*:: Guest >::FittedTrendModel > (arg0) }
                        #[export_name =
                        "augurs:core/types#[method]fitted-trend-model.predict"] unsafe
                        extern "C" fn export_method_fitted_trend_model_predict(arg0 : *
                        mut u8, arg1 : i32, arg2 : i32, arg3 : f64,) -> * mut u8 {
                        $($path_to_types)*::
                        _export_method_fitted_trend_model_predict_cabi::<<$ty as
                        $($path_to_types)*:: Guest >::FittedTrendModel > (arg0, arg1,
                        arg2, arg3) } #[export_name =
                        "cabi_post_augurs:core/types#[method]fitted-trend-model.predict"]
                        unsafe extern "C" fn
                        _post_return_method_fitted_trend_model_predict(arg0 : * mut u8,)
                        { $($path_to_types)*::
                        __post_return_method_fitted_trend_model_predict::<<$ty as
                        $($path_to_types)*:: Guest >::FittedTrendModel > (arg0) }
                        #[export_name = "augurs:core/types#[method]trend-model.fit"]
                        unsafe extern "C" fn export_method_trend_model_fit(arg0 : * mut
                        u8, arg1 : * mut u8, arg2 : usize,) -> * mut u8 {
                        $($path_to_types)*:: _export_method_trend_model_fit_cabi::<<$ty
                        as $($path_to_types)*:: Guest >::TrendModel > (arg0, arg1, arg2)
                        } const _ : () = { #[doc(hidden)] #[export_name =
                        "augurs:core/types#[dtor]fitted-trend-model"]
                        #[allow(non_snake_case)] unsafe extern "C" fn dtor(rep : * mut
                        u8) { $($path_to_types)*:: FittedTrendModel::dtor::< <$ty as
                        $($path_to_types)*:: Guest >::FittedTrendModel > (rep) } }; const
                        _ : () = { #[doc(hidden)] #[export_name =
                        "augurs:core/types#[dtor]trend-model"] #[allow(non_snake_case)]
                        unsafe extern "C" fn dtor(rep : * mut u8) { $($path_to_types)*::
                        TrendModel::dtor::< <$ty as $($path_to_types)*:: Guest
                        >::TrendModel > (rep) } }; };
                    };
                }
                #[doc(hidden)]
                pub(crate) use __export_augurs_core_types_cabi;
                #[repr(align(8))]
                struct _RetArea([::core::mem::MaybeUninit<u8>; 48]);
                static mut _RET_AREA: _RetArea = _RetArea(
                    [::core::mem::MaybeUninit::uninit(); 48],
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
    extern crate alloc as alloc_crate;
    pub use alloc_crate::alloc;
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
        exports::augurs::core::types::__export_augurs_core_types_cabi!($ty with_types_in
        $($path_to_types_root)*:: exports::augurs::core::types);
    };
}
#[doc(inline)]
pub(crate) use __export_component_impl as export;
#[cfg(target_arch = "wasm32")]
#[link_section = "component-type:wit-bindgen:0.35.0:augurs:core:component:encoded world"]
#[doc(hidden)]
pub static __WIT_BINDGEN_COMPONENT_TYPE: [u8; 596] = *b"\
\0asm\x0d\0\x01\0\0\x19\x16wit-component-encoding\x04\0\x07\xd4\x03\x01A\x02\x01\
A\x02\x01B\x1a\x01pu\x04\0\x0btime-series\x03\0\0\x01pu\x01p\x02\x04\0\x0fdistan\
ce-matrix\x03\0\x03\x01q\x01\x17invalid-distance-matrix\0\0\x04\0\x05error\x03\0\
\x05\x01r\x03\x05levelu\x05lower\x01\x05upper\x01\x04\0\x12forecast-intervals\x03\
\0\x07\x01k\x08\x01r\x02\x05point\x01\x09intervals\x09\x04\0\x08forecast\x03\0\x0a\
\x04\0\x12fitted-trend-model\x03\x01\x04\0\x0btrend-model\x03\x01\x01h\x0c\x01ku\
\x01j\x01\x0b\x01\x06\x01@\x02\x04self\x0e\x05level\x0f\0\x10\x04\0,[method]fitt\
ed-trend-model.predict-in-sample\x01\x11\x01@\x03\x04self\x0e\x07horizony\x05lev\
el\x0f\0\x10\x04\0\"[method]fitted-trend-model.predict\x01\x12\x01h\x0d\x01i\x0c\
\x01j\x01\x14\x01\x06\x01@\x02\x04self\x13\x01y\x01\0\x15\x04\0\x17[method]trend\
-model.fit\x01\x16\x04\0\x11augurs:core/types\x05\0\x04\0\x15augurs:core/compone\
nt\x04\0\x0b\x0f\x01\0\x09component\x03\0\0\0G\x09producers\x01\x0cprocessed-by\x02\
\x0dwit-component\x070.220.0\x10wit-bindgen-rust\x060.35.0";
#[inline(never)]
#[doc(hidden)]
pub fn __link_custom_section_describing_imports() {
    wit_bindgen_rt::maybe_link_cabi_realloc();
}
