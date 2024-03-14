//! Representation of basic types

/// Specification of data types
#[derive(Eq, PartialEq, Debug, Clone, Copy)]
#[repr(u8)]
pub enum DTYPE {
    /// 32 bit float
    Float32 = 0,
    /// 64 bit float
    Float64 = 1,
    /// 8 bit signed integer
    Int8 = 2,
    /// 32 bit signed integer
    Int32 = 3,
    /// 64 bit signed integer
    Int64 = 4,
    /// 8 bit unsigned integer
    Unsigned8 = 5,
    /// 32 bit unsigned integer
    Unsigned32 = 6,
    /// 64 bit unsigned integer
    Unsigned64 = 7,
    /// Machine dependent indexing type
    Usize = 8,
}

/// Mutability Property
#[derive(Eq, PartialEq, Debug, Clone, Copy)]
#[repr(u8)]
pub enum MUTABILITY {
    /// Not mutable
    NotMutable = 0,
    /// Mutable
    Mutable = 1,
}

/// Ownership Property
#[derive(Eq, PartialEq, Debug, Clone, Copy)]
#[repr(u8)]
pub enum OWNERSHIP {
    /// Not owner
    NotOwner = 0,
    /// Owner
    Owner = 1,
}

/// Conversion type
pub trait ConversionType: 'static {
    /// Dtype
    const D: DTYPE;
    /// Size
    const SIZE: usize;
}

impl ConversionType for f32 {
    const D: DTYPE = DTYPE::Float32;
    const SIZE: usize = 4;
}
impl ConversionType for f64 {
    const D: DTYPE = DTYPE::Float64;
    const SIZE: usize = 8;
}
impl ConversionType for i8 {
    const D: DTYPE = DTYPE::Int8;
    const SIZE: usize = 1;
}
impl ConversionType for i32 {
    const D: DTYPE = DTYPE::Int32;
    const SIZE: usize = 4;
}
impl ConversionType for i64 {
    const D: DTYPE = DTYPE::Int64;
    const SIZE: usize = 8;
}
impl ConversionType for u8 {
    const D: DTYPE = DTYPE::Unsigned8;
    const SIZE: usize = 1;
}
impl ConversionType for u32 {
    const D: DTYPE = DTYPE::Unsigned32;
    const SIZE: usize = 4;
}
impl ConversionType for u64 {
    const D: DTYPE = DTYPE::Unsigned64;
    const SIZE: usize = 8;
}

impl ConversionType for usize {
    const D: DTYPE = DTYPE::Usize;
    const SIZE: usize = 8;
}

/// Get dtype
pub fn get_dtype<T: ConversionType>() -> DTYPE {
    T::D
}

/// Assert dtype
pub fn assert_dtype<T: ConversionType>(d: DTYPE) {
    assert_eq!(get_dtype::<T>(), d);
}

/// Get size
pub fn get_size<T: ConversionType>() -> usize {
    T::SIZE
}

/// Get itemsize
pub fn get_itemsize(dtype: DTYPE) -> usize {
    match dtype {
        DTYPE::Float32 => crate::get_size::<f32>(),
        DTYPE::Float64 => crate::get_size::<f64>(),
        DTYPE::Unsigned8 => crate::get_size::<u8>(),
        DTYPE::Unsigned32 => crate::get_size::<u32>(),
        DTYPE::Unsigned64 => crate::get_size::<u64>(),
        DTYPE::Int8 => crate::get_size::<i8>(),
        DTYPE::Int32 => crate::get_size::<i32>(),
        DTYPE::Int64 => crate::get_size::<i64>(),
        DTYPE::Usize => crate::get_size::<usize>(),
    }
}
