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
            pub type TimeSeriesResult = _rt::Vec<f64>;
            /// A sequence of floats comprising a time series.
            pub type TimeSeriesParam<'a> = &'a [f64];
            #[derive(Clone)]
            pub enum Error {
                /// The distance matrix is not square.
                InvalidDistanceMatrix,
                /// An error occurred while fitting the model.
                Fit(_rt::String),
                /// An error occurred while predicting with the model.
                Predict(_rt::String),
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
                        Error::Fit(e) => f.debug_tuple("Error::Fit").field(e).finish(),
                        Error::Predict(e) => {
                            f.debug_tuple("Error::Predict").field(e).finish()
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
                pub lower: TimeSeriesResult,
                pub upper: TimeSeriesResult,
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
                pub point: TimeSeriesResult,
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
            impl FittedTrendModel {
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
            }
            unsafe impl _rt::WasmResource for FittedTrendModel {
                #[inline]
                unsafe fn drop(_handle: u32) {
                    #[cfg(not(target_arch = "wasm32"))]
                    unreachable!();
                    #[cfg(target_arch = "wasm32")]
                    {
                        #[link(wasm_import_module = "augurs:core/types")]
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
            impl TrendModel {
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
            }
            unsafe impl _rt::WasmResource for TrendModel {
                #[inline]
                unsafe fn drop(_handle: u32) {
                    #[cfg(not(target_arch = "wasm32"))]
                    unreachable!();
                    #[cfg(target_arch = "wasm32")]
                    {
                        #[link(wasm_import_module = "augurs:core/types")]
                        extern "C" {
                            #[link_name = "[resource-drop]trend-model"]
                            fn drop(_: u32);
                        }
                        drop(_handle);
                    }
                }
            }
            impl FittedTrendModel {
                #[allow(unused_unsafe, clippy::all)]
                pub fn predict_in_sample(
                    &self,
                    level: Option<f64>,
                ) -> Result<Forecast, Error> {
                    unsafe {
                        #[repr(align(8))]
                        struct RetArea([::core::mem::MaybeUninit<u8>; 48]);
                        let mut ret_area = RetArea(
                            [::core::mem::MaybeUninit::uninit(); 48],
                        );
                        let (result0_0, result0_1) = match level {
                            Some(e) => (1i32, _rt::as_f64(e)),
                            None => (0i32, 0.0f64),
                        };
                        let ptr1 = ret_area.0.as_mut_ptr().cast::<u8>();
                        #[cfg(target_arch = "wasm32")]
                        #[link(wasm_import_module = "augurs:core/types")]
                        extern "C" {
                            #[link_name = "[method]fitted-trend-model.predict-in-sample"]
                            fn wit_import(_: i32, _: i32, _: f64, _: *mut u8);
                        }
                        #[cfg(not(target_arch = "wasm32"))]
                        fn wit_import(_: i32, _: i32, _: f64, _: *mut u8) {
                            unreachable!()
                        }
                        wit_import((self).handle() as i32, result0_0, result0_1, ptr1);
                        let l2 = i32::from(*ptr1.add(0).cast::<u8>());
                        match l2 {
                            0 => {
                                let e = {
                                    let l3 = *ptr1.add(8).cast::<*mut u8>();
                                    let l4 = *ptr1.add(12).cast::<usize>();
                                    let len5 = l4;
                                    let l6 = i32::from(*ptr1.add(16).cast::<u8>());
                                    Forecast {
                                        point: _rt::Vec::from_raw_parts(l3.cast(), len5, len5),
                                        intervals: match l6 {
                                            0 => None,
                                            1 => {
                                                let e = {
                                                    let l7 = *ptr1.add(24).cast::<f64>();
                                                    let l8 = *ptr1.add(32).cast::<*mut u8>();
                                                    let l9 = *ptr1.add(36).cast::<usize>();
                                                    let len10 = l9;
                                                    let l11 = *ptr1.add(40).cast::<*mut u8>();
                                                    let l12 = *ptr1.add(44).cast::<usize>();
                                                    let len13 = l12;
                                                    ForecastIntervals {
                                                        level: l7,
                                                        lower: _rt::Vec::from_raw_parts(l8.cast(), len10, len10),
                                                        upper: _rt::Vec::from_raw_parts(l11.cast(), len13, len13),
                                                    }
                                                };
                                                Some(e)
                                            }
                                            _ => _rt::invalid_enum_discriminant(),
                                        },
                                    }
                                };
                                Ok(e)
                            }
                            1 => {
                                let e = {
                                    let l14 = i32::from(*ptr1.add(8).cast::<u8>());
                                    let v21 = match l14 {
                                        0 => Error::InvalidDistanceMatrix,
                                        1 => {
                                            let e21 = {
                                                let l15 = *ptr1.add(12).cast::<*mut u8>();
                                                let l16 = *ptr1.add(16).cast::<usize>();
                                                let len17 = l16;
                                                let bytes17 = _rt::Vec::from_raw_parts(
                                                    l15.cast(),
                                                    len17,
                                                    len17,
                                                );
                                                _rt::string_lift(bytes17)
                                            };
                                            Error::Fit(e21)
                                        }
                                        n => {
                                            debug_assert_eq!(n, 2, "invalid enum discriminant");
                                            let e21 = {
                                                let l18 = *ptr1.add(12).cast::<*mut u8>();
                                                let l19 = *ptr1.add(16).cast::<usize>();
                                                let len20 = l19;
                                                let bytes20 = _rt::Vec::from_raw_parts(
                                                    l18.cast(),
                                                    len20,
                                                    len20,
                                                );
                                                _rt::string_lift(bytes20)
                                            };
                                            Error::Predict(e21)
                                        }
                                    };
                                    v21
                                };
                                Err(e)
                            }
                            _ => _rt::invalid_enum_discriminant(),
                        }
                    }
                }
            }
            impl FittedTrendModel {
                #[allow(unused_unsafe, clippy::all)]
                pub fn predict(
                    &self,
                    horizon: u32,
                    level: Option<f64>,
                ) -> Result<Forecast, Error> {
                    unsafe {
                        #[repr(align(8))]
                        struct RetArea([::core::mem::MaybeUninit<u8>; 48]);
                        let mut ret_area = RetArea(
                            [::core::mem::MaybeUninit::uninit(); 48],
                        );
                        let (result0_0, result0_1) = match level {
                            Some(e) => (1i32, _rt::as_f64(e)),
                            None => (0i32, 0.0f64),
                        };
                        let ptr1 = ret_area.0.as_mut_ptr().cast::<u8>();
                        #[cfg(target_arch = "wasm32")]
                        #[link(wasm_import_module = "augurs:core/types")]
                        extern "C" {
                            #[link_name = "[method]fitted-trend-model.predict"]
                            fn wit_import(_: i32, _: i32, _: i32, _: f64, _: *mut u8);
                        }
                        #[cfg(not(target_arch = "wasm32"))]
                        fn wit_import(_: i32, _: i32, _: i32, _: f64, _: *mut u8) {
                            unreachable!()
                        }
                        wit_import(
                            (self).handle() as i32,
                            _rt::as_i32(&horizon),
                            result0_0,
                            result0_1,
                            ptr1,
                        );
                        let l2 = i32::from(*ptr1.add(0).cast::<u8>());
                        match l2 {
                            0 => {
                                let e = {
                                    let l3 = *ptr1.add(8).cast::<*mut u8>();
                                    let l4 = *ptr1.add(12).cast::<usize>();
                                    let len5 = l4;
                                    let l6 = i32::from(*ptr1.add(16).cast::<u8>());
                                    Forecast {
                                        point: _rt::Vec::from_raw_parts(l3.cast(), len5, len5),
                                        intervals: match l6 {
                                            0 => None,
                                            1 => {
                                                let e = {
                                                    let l7 = *ptr1.add(24).cast::<f64>();
                                                    let l8 = *ptr1.add(32).cast::<*mut u8>();
                                                    let l9 = *ptr1.add(36).cast::<usize>();
                                                    let len10 = l9;
                                                    let l11 = *ptr1.add(40).cast::<*mut u8>();
                                                    let l12 = *ptr1.add(44).cast::<usize>();
                                                    let len13 = l12;
                                                    ForecastIntervals {
                                                        level: l7,
                                                        lower: _rt::Vec::from_raw_parts(l8.cast(), len10, len10),
                                                        upper: _rt::Vec::from_raw_parts(l11.cast(), len13, len13),
                                                    }
                                                };
                                                Some(e)
                                            }
                                            _ => _rt::invalid_enum_discriminant(),
                                        },
                                    }
                                };
                                Ok(e)
                            }
                            1 => {
                                let e = {
                                    let l14 = i32::from(*ptr1.add(8).cast::<u8>());
                                    let v21 = match l14 {
                                        0 => Error::InvalidDistanceMatrix,
                                        1 => {
                                            let e21 = {
                                                let l15 = *ptr1.add(12).cast::<*mut u8>();
                                                let l16 = *ptr1.add(16).cast::<usize>();
                                                let len17 = l16;
                                                let bytes17 = _rt::Vec::from_raw_parts(
                                                    l15.cast(),
                                                    len17,
                                                    len17,
                                                );
                                                _rt::string_lift(bytes17)
                                            };
                                            Error::Fit(e21)
                                        }
                                        n => {
                                            debug_assert_eq!(n, 2, "invalid enum discriminant");
                                            let e21 = {
                                                let l18 = *ptr1.add(12).cast::<*mut u8>();
                                                let l19 = *ptr1.add(16).cast::<usize>();
                                                let len20 = l19;
                                                let bytes20 = _rt::Vec::from_raw_parts(
                                                    l18.cast(),
                                                    len20,
                                                    len20,
                                                );
                                                _rt::string_lift(bytes20)
                                            };
                                            Error::Predict(e21)
                                        }
                                    };
                                    v21
                                };
                                Err(e)
                            }
                            _ => _rt::invalid_enum_discriminant(),
                        }
                    }
                }
            }
            impl TrendModel {
                #[allow(unused_unsafe, clippy::all)]
                pub fn fit(
                    &self,
                    y: TimeSeriesParam<'_>,
                ) -> Result<FittedTrendModel, Error> {
                    unsafe {
                        #[repr(align(4))]
                        struct RetArea([::core::mem::MaybeUninit<u8>; 16]);
                        let mut ret_area = RetArea(
                            [::core::mem::MaybeUninit::uninit(); 16],
                        );
                        let vec0 = y;
                        let ptr0 = vec0.as_ptr().cast::<u8>();
                        let len0 = vec0.len();
                        let ptr1 = ret_area.0.as_mut_ptr().cast::<u8>();
                        #[cfg(target_arch = "wasm32")]
                        #[link(wasm_import_module = "augurs:core/types")]
                        extern "C" {
                            #[link_name = "[method]trend-model.fit"]
                            fn wit_import(_: i32, _: *mut u8, _: usize, _: *mut u8);
                        }
                        #[cfg(not(target_arch = "wasm32"))]
                        fn wit_import(_: i32, _: *mut u8, _: usize, _: *mut u8) {
                            unreachable!()
                        }
                        wit_import((self).handle() as i32, ptr0.cast_mut(), len0, ptr1);
                        let l2 = i32::from(*ptr1.add(0).cast::<u8>());
                        match l2 {
                            0 => {
                                let e = {
                                    let l3 = *ptr1.add(4).cast::<i32>();
                                    FittedTrendModel::from_handle(l3 as u32)
                                };
                                Ok(e)
                            }
                            1 => {
                                let e = {
                                    let l4 = i32::from(*ptr1.add(4).cast::<u8>());
                                    let v11 = match l4 {
                                        0 => Error::InvalidDistanceMatrix,
                                        1 => {
                                            let e11 = {
                                                let l5 = *ptr1.add(8).cast::<*mut u8>();
                                                let l6 = *ptr1.add(12).cast::<usize>();
                                                let len7 = l6;
                                                let bytes7 = _rt::Vec::from_raw_parts(
                                                    l5.cast(),
                                                    len7,
                                                    len7,
                                                );
                                                _rt::string_lift(bytes7)
                                            };
                                            Error::Fit(e11)
                                        }
                                        n => {
                                            debug_assert_eq!(n, 2, "invalid enum discriminant");
                                            let e11 = {
                                                let l8 = *ptr1.add(8).cast::<*mut u8>();
                                                let l9 = *ptr1.add(12).cast::<usize>();
                                                let len10 = l9;
                                                let bytes10 = _rt::Vec::from_raw_parts(
                                                    l8.cast(),
                                                    len10,
                                                    len10,
                                                );
                                                _rt::string_lift(bytes10)
                                            };
                                            Error::Predict(e11)
                                        }
                                    };
                                    v11
                                };
                                Err(e)
                            }
                            _ => _rt::invalid_enum_discriminant(),
                        }
                    }
                }
            }
        }
    }
}
#[allow(dead_code)]
pub mod exports {
    #[allow(dead_code)]
    pub mod augurs {
        #[allow(dead_code)]
        pub mod mstl {
            #[allow(dead_code, clippy::all)]
            pub mod mstl {
                #[used]
                #[doc(hidden)]
                static __FORCE_SECTION_REF: fn() = super::super::super::super::__link_custom_section_describing_imports;
                use super::super::super::super::_rt;
                pub type Error = super::super::super::super::augurs::core::types::Error;
                pub type Forecast = super::super::super::super::augurs::core::types::Forecast;
                pub type TimeSeries = super::super::super::super::augurs::core::types::TimeSeriesResult;
                pub type TrendModel = super::super::super::super::augurs::core::types::TrendModel;
                #[derive(Debug)]
                #[repr(transparent)]
                pub struct Mstl {
                    handle: _rt::Resource<Mstl>,
                }
                type _MstlRep<T> = Option<T>;
                impl Mstl {
                    /// Creates a new resource from the specified representation.
                    ///
                    /// This function will create a new resource handle by moving `val` onto
                    /// the heap and then passing that heap pointer to the component model to
                    /// create a handle. The owned handle is then returned as `Mstl`.
                    pub fn new<T: GuestMstl>(val: T) -> Self {
                        Self::type_guard::<T>();
                        let val: _MstlRep<T> = Some(val);
                        let ptr: *mut _MstlRep<T> = _rt::Box::into_raw(
                            _rt::Box::new(val),
                        );
                        unsafe { Self::from_handle(T::_resource_new(ptr.cast())) }
                    }
                    /// Gets access to the underlying `T` which represents this resource.
                    pub fn get<T: GuestMstl>(&self) -> &T {
                        let ptr = unsafe { &*self.as_ptr::<T>() };
                        ptr.as_ref().unwrap()
                    }
                    /// Gets mutable access to the underlying `T` which represents this
                    /// resource.
                    pub fn get_mut<T: GuestMstl>(&mut self) -> &mut T {
                        let ptr = unsafe { &mut *self.as_ptr::<T>() };
                        ptr.as_mut().unwrap()
                    }
                    /// Consumes this resource and returns the underlying `T`.
                    pub fn into_inner<T: GuestMstl>(self) -> T {
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
                        let _ = _rt::Box::from_raw(handle as *mut _MstlRep<T>);
                    }
                    fn as_ptr<T: GuestMstl>(&self) -> *mut _MstlRep<T> {
                        Mstl::type_guard::<T>();
                        T::_resource_rep(self.handle()).cast()
                    }
                }
                /// A borrowed version of [`Mstl`] which represents a borrowed value
                /// with the lifetime `'a`.
                #[derive(Debug)]
                #[repr(transparent)]
                pub struct MstlBorrow<'a> {
                    rep: *mut u8,
                    _marker: core::marker::PhantomData<&'a Mstl>,
                }
                impl<'a> MstlBorrow<'a> {
                    #[doc(hidden)]
                    pub unsafe fn lift(rep: usize) -> Self {
                        Self {
                            rep: rep as *mut u8,
                            _marker: core::marker::PhantomData,
                        }
                    }
                    /// Gets access to the underlying `T` in this resource.
                    pub fn get<T: GuestMstl>(&self) -> &T {
                        let ptr = unsafe { &mut *self.as_ptr::<T>() };
                        ptr.as_ref().unwrap()
                    }
                    fn as_ptr<T: 'static>(&self) -> *mut _MstlRep<T> {
                        Mstl::type_guard::<T>();
                        self.rep.cast()
                    }
                }
                unsafe impl _rt::WasmResource for Mstl {
                    #[inline]
                    unsafe fn drop(_handle: u32) {
                        #[cfg(not(target_arch = "wasm32"))]
                        unreachable!();
                        #[cfg(target_arch = "wasm32")]
                        {
                            #[link(wasm_import_module = "[export]augurs:mstl/mstl")]
                            extern "C" {
                                #[link_name = "[resource-drop]mstl"]
                                fn drop(_: u32);
                            }
                            drop(_handle);
                        }
                    }
                }
                #[derive(Debug)]
                #[repr(transparent)]
                pub struct FittedMstl {
                    handle: _rt::Resource<FittedMstl>,
                }
                type _FittedMstlRep<T> = Option<T>;
                impl FittedMstl {
                    /// Creates a new resource from the specified representation.
                    ///
                    /// This function will create a new resource handle by moving `val` onto
                    /// the heap and then passing that heap pointer to the component model to
                    /// create a handle. The owned handle is then returned as `FittedMstl`.
                    pub fn new<T: GuestFittedMstl>(val: T) -> Self {
                        Self::type_guard::<T>();
                        let val: _FittedMstlRep<T> = Some(val);
                        let ptr: *mut _FittedMstlRep<T> = _rt::Box::into_raw(
                            _rt::Box::new(val),
                        );
                        unsafe { Self::from_handle(T::_resource_new(ptr.cast())) }
                    }
                    /// Gets access to the underlying `T` which represents this resource.
                    pub fn get<T: GuestFittedMstl>(&self) -> &T {
                        let ptr = unsafe { &*self.as_ptr::<T>() };
                        ptr.as_ref().unwrap()
                    }
                    /// Gets mutable access to the underlying `T` which represents this
                    /// resource.
                    pub fn get_mut<T: GuestFittedMstl>(&mut self) -> &mut T {
                        let ptr = unsafe { &mut *self.as_ptr::<T>() };
                        ptr.as_mut().unwrap()
                    }
                    /// Consumes this resource and returns the underlying `T`.
                    pub fn into_inner<T: GuestFittedMstl>(self) -> T {
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
                        let _ = _rt::Box::from_raw(handle as *mut _FittedMstlRep<T>);
                    }
                    fn as_ptr<T: GuestFittedMstl>(&self) -> *mut _FittedMstlRep<T> {
                        FittedMstl::type_guard::<T>();
                        T::_resource_rep(self.handle()).cast()
                    }
                }
                /// A borrowed version of [`FittedMstl`] which represents a borrowed value
                /// with the lifetime `'a`.
                #[derive(Debug)]
                #[repr(transparent)]
                pub struct FittedMstlBorrow<'a> {
                    rep: *mut u8,
                    _marker: core::marker::PhantomData<&'a FittedMstl>,
                }
                impl<'a> FittedMstlBorrow<'a> {
                    #[doc(hidden)]
                    pub unsafe fn lift(rep: usize) -> Self {
                        Self {
                            rep: rep as *mut u8,
                            _marker: core::marker::PhantomData,
                        }
                    }
                    /// Gets access to the underlying `T` in this resource.
                    pub fn get<T: GuestFittedMstl>(&self) -> &T {
                        let ptr = unsafe { &mut *self.as_ptr::<T>() };
                        ptr.as_ref().unwrap()
                    }
                    fn as_ptr<T: 'static>(&self) -> *mut _FittedMstlRep<T> {
                        FittedMstl::type_guard::<T>();
                        self.rep.cast()
                    }
                }
                unsafe impl _rt::WasmResource for FittedMstl {
                    #[inline]
                    unsafe fn drop(_handle: u32) {
                        #[cfg(not(target_arch = "wasm32"))]
                        unreachable!();
                        #[cfg(target_arch = "wasm32")]
                        {
                            #[link(wasm_import_module = "[export]augurs:mstl/mstl")]
                            extern "C" {
                                #[link_name = "[resource-drop]fitted-mstl"]
                                fn drop(_: u32);
                            }
                            drop(_handle);
                        }
                    }
                }
                #[doc(hidden)]
                #[allow(non_snake_case)]
                pub unsafe fn _export_constructor_mstl_cabi<T: GuestMstl>(
                    arg0: *mut u8,
                    arg1: usize,
                    arg2: i32,
                ) -> i32 {
                    #[cfg(target_arch = "wasm32")] _rt::run_ctors_once();
                    let len0 = arg1;
                    let result1 = Mstl::new(
                        T::new(
                            _rt::Vec::from_raw_parts(arg0.cast(), len0, len0),
                            super::super::super::super::augurs::core::types::TrendModel::from_handle(
                                arg2 as u32,
                            ),
                        ),
                    );
                    (result1).take_handle() as i32
                }
                #[doc(hidden)]
                #[allow(non_snake_case)]
                pub unsafe fn _export_method_mstl_fit_cabi<T: GuestMstl>(
                    arg0: *mut u8,
                    arg1: *mut u8,
                    arg2: usize,
                ) -> *mut u8 {
                    #[cfg(target_arch = "wasm32")] _rt::run_ctors_once();
                    let len0 = arg2;
                    let result1 = T::fit(
                        MstlBorrow::lift(arg0 as u32 as usize).get(),
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
                            use super::super::super::super::augurs::core::types::Error as V5;
                            match e {
                                V5::InvalidDistanceMatrix => {
                                    *ptr2.add(4).cast::<u8>() = (0i32) as u8;
                                }
                                V5::Fit(e) => {
                                    *ptr2.add(4).cast::<u8>() = (1i32) as u8;
                                    let vec3 = (e.into_bytes()).into_boxed_slice();
                                    let ptr3 = vec3.as_ptr().cast::<u8>();
                                    let len3 = vec3.len();
                                    ::core::mem::forget(vec3);
                                    *ptr2.add(12).cast::<usize>() = len3;
                                    *ptr2.add(8).cast::<*mut u8>() = ptr3.cast_mut();
                                }
                                V5::Predict(e) => {
                                    *ptr2.add(4).cast::<u8>() = (2i32) as u8;
                                    let vec4 = (e.into_bytes()).into_boxed_slice();
                                    let ptr4 = vec4.as_ptr().cast::<u8>();
                                    let len4 = vec4.len();
                                    ::core::mem::forget(vec4);
                                    *ptr2.add(12).cast::<usize>() = len4;
                                    *ptr2.add(8).cast::<*mut u8>() = ptr4.cast_mut();
                                }
                            }
                        }
                    };
                    ptr2
                }
                #[doc(hidden)]
                #[allow(non_snake_case)]
                pub unsafe fn __post_return_method_mstl_fit<T: GuestMstl>(
                    arg0: *mut u8,
                ) {
                    let l0 = i32::from(*arg0.add(0).cast::<u8>());
                    match l0 {
                        0 => {}
                        _ => {
                            let l1 = i32::from(*arg0.add(4).cast::<u8>());
                            match l1 {
                                0 => {}
                                1 => {
                                    let l2 = *arg0.add(8).cast::<*mut u8>();
                                    let l3 = *arg0.add(12).cast::<usize>();
                                    _rt::cabi_dealloc(l2, l3, 1);
                                }
                                _ => {
                                    let l4 = *arg0.add(8).cast::<*mut u8>();
                                    let l5 = *arg0.add(12).cast::<usize>();
                                    _rt::cabi_dealloc(l4, l5, 1);
                                }
                            }
                        }
                    }
                }
                #[doc(hidden)]
                #[allow(non_snake_case)]
                pub unsafe fn _export_method_fitted_mstl_predict_cabi<
                    T: GuestFittedMstl,
                >(arg0: *mut u8, arg1: i32, arg2: i32, arg3: f64) -> *mut u8 {
                    #[cfg(target_arch = "wasm32")] _rt::run_ctors_once();
                    let result0 = T::predict(
                        FittedMstlBorrow::lift(arg0 as u32 as usize).get(),
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
                            let super::super::super::super::augurs::core::types::Forecast {
                                point: point2,
                                intervals: intervals2,
                            } = e;
                            let vec3 = (point2).into_boxed_slice();
                            let ptr3 = vec3.as_ptr().cast::<u8>();
                            let len3 = vec3.len();
                            ::core::mem::forget(vec3);
                            *ptr1.add(12).cast::<usize>() = len3;
                            *ptr1.add(8).cast::<*mut u8>() = ptr3.cast_mut();
                            match intervals2 {
                                Some(e) => {
                                    *ptr1.add(16).cast::<u8>() = (1i32) as u8;
                                    let super::super::super::super::augurs::core::types::ForecastIntervals {
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
                            use super::super::super::super::augurs::core::types::Error as V9;
                            match e {
                                V9::InvalidDistanceMatrix => {
                                    *ptr1.add(8).cast::<u8>() = (0i32) as u8;
                                }
                                V9::Fit(e) => {
                                    *ptr1.add(8).cast::<u8>() = (1i32) as u8;
                                    let vec7 = (e.into_bytes()).into_boxed_slice();
                                    let ptr7 = vec7.as_ptr().cast::<u8>();
                                    let len7 = vec7.len();
                                    ::core::mem::forget(vec7);
                                    *ptr1.add(16).cast::<usize>() = len7;
                                    *ptr1.add(12).cast::<*mut u8>() = ptr7.cast_mut();
                                }
                                V9::Predict(e) => {
                                    *ptr1.add(8).cast::<u8>() = (2i32) as u8;
                                    let vec8 = (e.into_bytes()).into_boxed_slice();
                                    let ptr8 = vec8.as_ptr().cast::<u8>();
                                    let len8 = vec8.len();
                                    ::core::mem::forget(vec8);
                                    *ptr1.add(16).cast::<usize>() = len8;
                                    *ptr1.add(12).cast::<*mut u8>() = ptr8.cast_mut();
                                }
                            }
                        }
                    };
                    ptr1
                }
                #[doc(hidden)]
                #[allow(non_snake_case)]
                pub unsafe fn __post_return_method_fitted_mstl_predict<
                    T: GuestFittedMstl,
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
                        _ => {
                            let l11 = i32::from(*arg0.add(8).cast::<u8>());
                            match l11 {
                                0 => {}
                                1 => {
                                    let l12 = *arg0.add(12).cast::<*mut u8>();
                                    let l13 = *arg0.add(16).cast::<usize>();
                                    _rt::cabi_dealloc(l12, l13, 1);
                                }
                                _ => {
                                    let l14 = *arg0.add(12).cast::<*mut u8>();
                                    let l15 = *arg0.add(16).cast::<usize>();
                                    _rt::cabi_dealloc(l14, l15, 1);
                                }
                            }
                        }
                    }
                }
                pub trait Guest {
                    type Mstl: GuestMstl;
                    type FittedMstl: GuestFittedMstl;
                }
                pub trait GuestMstl: 'static {
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
                            #[link(wasm_import_module = "[export]augurs:mstl/mstl")]
                            extern "C" {
                                #[link_name = "[resource-new]mstl"]
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
                            #[link(wasm_import_module = "[export]augurs:mstl/mstl")]
                            extern "C" {
                                #[link_name = "[resource-rep]mstl"]
                                fn rep(_: u32) -> *mut u8;
                            }
                            unsafe { rep(handle) }
                        }
                    }
                    /// /// Create a new MSTL model with an ETS trend model.
                    /// ets: static func(periods: list<u32>, opts: option<ets-opts>) -> mstl;
                    fn new(periods: _rt::Vec<u32>, trend_model: TrendModel) -> Self;
                    fn fit(&self, y: TimeSeries) -> Result<FittedMstl, Error>;
                }
                pub trait GuestFittedMstl: 'static {
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
                            #[link(wasm_import_module = "[export]augurs:mstl/mstl")]
                            extern "C" {
                                #[link_name = "[resource-new]fitted-mstl"]
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
                            #[link(wasm_import_module = "[export]augurs:mstl/mstl")]
                            extern "C" {
                                #[link_name = "[resource-rep]fitted-mstl"]
                                fn rep(_: u32) -> *mut u8;
                            }
                            unsafe { rep(handle) }
                        }
                    }
                    fn predict(
                        &self,
                        horizon: u32,
                        level: Option<f64>,
                    ) -> Result<Forecast, Error>;
                }
                #[doc(hidden)]
                macro_rules! __export_augurs_mstl_mstl_cabi {
                    ($ty:ident with_types_in $($path_to_types:tt)*) => {
                        const _ : () = { #[export_name =
                        "augurs:mstl/mstl#[constructor]mstl"] unsafe extern "C" fn
                        export_constructor_mstl(arg0 : * mut u8, arg1 : usize, arg2 :
                        i32,) -> i32 { $($path_to_types)*::
                        _export_constructor_mstl_cabi::<<$ty as $($path_to_types)*::
                        Guest >::Mstl > (arg0, arg1, arg2) } #[export_name =
                        "augurs:mstl/mstl#[method]mstl.fit"] unsafe extern "C" fn
                        export_method_mstl_fit(arg0 : * mut u8, arg1 : * mut u8, arg2 :
                        usize,) -> * mut u8 { $($path_to_types)*::
                        _export_method_mstl_fit_cabi::<<$ty as $($path_to_types)*:: Guest
                        >::Mstl > (arg0, arg1, arg2) } #[export_name =
                        "cabi_post_augurs:mstl/mstl#[method]mstl.fit"] unsafe extern "C"
                        fn _post_return_method_mstl_fit(arg0 : * mut u8,) {
                        $($path_to_types)*:: __post_return_method_mstl_fit::<<$ty as
                        $($path_to_types)*:: Guest >::Mstl > (arg0) } #[export_name =
                        "augurs:mstl/mstl#[method]fitted-mstl.predict"] unsafe extern "C"
                        fn export_method_fitted_mstl_predict(arg0 : * mut u8, arg1 : i32,
                        arg2 : i32, arg3 : f64,) -> * mut u8 { $($path_to_types)*::
                        _export_method_fitted_mstl_predict_cabi::<<$ty as
                        $($path_to_types)*:: Guest >::FittedMstl > (arg0, arg1, arg2,
                        arg3) } #[export_name =
                        "cabi_post_augurs:mstl/mstl#[method]fitted-mstl.predict"] unsafe
                        extern "C" fn _post_return_method_fitted_mstl_predict(arg0 : *
                        mut u8,) { $($path_to_types)*::
                        __post_return_method_fitted_mstl_predict::<<$ty as
                        $($path_to_types)*:: Guest >::FittedMstl > (arg0) } const _ : ()
                        = { #[doc(hidden)] #[export_name = "augurs:mstl/mstl#[dtor]mstl"]
                        #[allow(non_snake_case)] unsafe extern "C" fn dtor(rep : * mut
                        u8) { $($path_to_types)*:: Mstl::dtor::< <$ty as
                        $($path_to_types)*:: Guest >::Mstl > (rep) } }; const _ : () = {
                        #[doc(hidden)] #[export_name =
                        "augurs:mstl/mstl#[dtor]fitted-mstl"] #[allow(non_snake_case)]
                        unsafe extern "C" fn dtor(rep : * mut u8) { $($path_to_types)*::
                        FittedMstl::dtor::< <$ty as $($path_to_types)*:: Guest
                        >::FittedMstl > (rep) } }; };
                    };
                }
                #[doc(hidden)]
                pub(crate) use __export_augurs_mstl_mstl_cabi;
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
    pub use alloc_crate::string::String;
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
    pub unsafe fn invalid_enum_discriminant<T>() -> T {
        if cfg!(debug_assertions) {
            panic!("invalid enum discriminant")
        } else {
            core::hint::unreachable_unchecked()
        }
    }
    pub unsafe fn string_lift(bytes: Vec<u8>) -> String {
        if cfg!(debug_assertions) {
            String::from_utf8(bytes).unwrap()
        } else {
            String::from_utf8_unchecked(bytes)
        }
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
        exports::augurs::mstl::mstl::__export_augurs_mstl_mstl_cabi!($ty with_types_in
        $($path_to_types_root)*:: exports::augurs::mstl::mstl);
    };
}
#[doc(inline)]
pub(crate) use __export_component_impl as export;
#[cfg(target_arch = "wasm32")]
#[link_section = "component-type:wit-bindgen:0.35.0:augurs:mstl:component:encoded world"]
#[doc(hidden)]
pub static __WIT_BINDGEN_COMPONENT_TYPE: [u8; 973] = *b"\
\0asm\x0d\0\x01\0\0\x19\x16wit-component-encoding\x04\0\x07\xcd\x06\x01A\x02\x01\
A\x08\x01B\x1a\x01pu\x04\0\x0btime-series\x03\0\0\x01pu\x01p\x02\x04\0\x0fdistan\
ce-matrix\x03\0\x03\x01q\x03\x17invalid-distance-matrix\0\0\x03fit\x01s\0\x07pre\
dict\x01s\0\x04\0\x05error\x03\0\x05\x01r\x03\x05levelu\x05lower\x01\x05upper\x01\
\x04\0\x12forecast-intervals\x03\0\x07\x01k\x08\x01r\x02\x05point\x01\x09interva\
ls\x09\x04\0\x08forecast\x03\0\x0a\x04\0\x12fitted-trend-model\x03\x01\x04\0\x0b\
trend-model\x03\x01\x01h\x0c\x01ku\x01j\x01\x0b\x01\x06\x01@\x02\x04self\x0e\x05\
level\x0f\0\x10\x04\0,[method]fitted-trend-model.predict-in-sample\x01\x11\x01@\x03\
\x04self\x0e\x07horizony\x05level\x0f\0\x10\x04\0\"[method]fitted-trend-model.pr\
edict\x01\x12\x01h\x0d\x01i\x0c\x01j\x01\x14\x01\x06\x01@\x02\x04self\x13\x01y\x01\
\0\x15\x04\0\x17[method]trend-model.fit\x01\x16\x03\0\x11augurs:core/types\x05\0\
\x02\x03\0\0\x05error\x02\x03\0\0\x08forecast\x02\x03\0\0\x0btime-series\x02\x03\
\0\0\x0btrend-model\x01B\x19\x02\x03\x02\x01\x01\x04\0\x05error\x03\0\0\x02\x03\x02\
\x01\x02\x04\0\x08forecast\x03\0\x02\x02\x03\x02\x01\x03\x04\0\x0btime-series\x03\
\0\x04\x02\x03\x02\x01\x04\x04\0\x0btrend-model\x03\0\x06\x04\0\x04mstl\x03\x01\x04\
\0\x0bfitted-mstl\x03\x01\x01py\x01i\x07\x01i\x08\x01@\x02\x07periods\x0a\x0btre\
nd-model\x0b\0\x0c\x04\0\x11[constructor]mstl\x01\x0d\x01h\x08\x01i\x09\x01j\x01\
\x0f\x01\x01\x01@\x02\x04self\x0e\x01y\x05\0\x10\x04\0\x10[method]mstl.fit\x01\x11\
\x01h\x09\x01ku\x01j\x01\x03\x01\x01\x01@\x03\x04self\x12\x07horizony\x05level\x13\
\0\x14\x04\0\x1b[method]fitted-mstl.predict\x01\x15\x04\0\x10augurs:mstl/mstl\x05\
\x05\x04\0\x15augurs:mstl/component\x04\0\x0b\x0f\x01\0\x09component\x03\0\0\0G\x09\
producers\x01\x0cprocessed-by\x02\x0dwit-component\x070.220.0\x10wit-bindgen-rus\
t\x060.35.0";
#[inline(never)]
#[doc(hidden)]
pub fn __link_custom_section_describing_imports() {
    wit_bindgen_rt::maybe_link_cabi_realloc();
}
