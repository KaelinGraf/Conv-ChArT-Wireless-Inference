#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};


#[link(name = "convchart_interfaces__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__convchart_interfaces__msg__InferenceResult() -> *const std::ffi::c_void;
}

#[link(name = "convchart_interfaces__rosidl_generator_c")]
extern "C" {
    fn convchart_interfaces__msg__InferenceResult__init(msg: *mut InferenceResult) -> bool;
    fn convchart_interfaces__msg__InferenceResult__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<InferenceResult>, size: usize) -> bool;
    fn convchart_interfaces__msg__InferenceResult__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<InferenceResult>);
    fn convchart_interfaces__msg__InferenceResult__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<InferenceResult>, out_seq: *mut rosidl_runtime_rs::Sequence<InferenceResult>) -> bool;
}

// Corresponds to convchart_interfaces__msg__InferenceResult
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]


// This struct is not documented.
#[allow(missing_docs)]

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct InferenceResult {

    // This member is not documented.
    #[allow(missing_docs)]
    pub header: std_msgs::msg::rmw::Header,


    // This member is not documented.
    #[allow(missing_docs)]
    pub pose: geometry_msgs::msg::rmw::PoseWithCovariance,

    /// RMS of pose result
    pub rms: f32,

    /// num of corners
    pub num_used: u8,

    /// if non-empty, then the inference result is invalid
    pub reason: rosidl_runtime_rs::String,

    /// if ambiguous, compute suprise
    pub ambiguous: bool,


    // This member is not documented.
    #[allow(missing_docs)]
    pub pose_alt: geometry_msgs::msg::rmw::PoseWithCovariance,

    /// RMS of pose result
    pub rms_alt: f32,

    /// num of corners
    pub num_used_alt: u8,

}



impl Default for InferenceResult {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !convchart_interfaces__msg__InferenceResult__init(&mut msg as *mut _) {
        panic!("Call to convchart_interfaces__msg__InferenceResult__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for InferenceResult {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { convchart_interfaces__msg__InferenceResult__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { convchart_interfaces__msg__InferenceResult__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { convchart_interfaces__msg__InferenceResult__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for InferenceResult {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for InferenceResult where Self: Sized {
  const TYPE_NAME: &'static str = "convchart_interfaces/msg/InferenceResult";
  fn get_type_support() -> *const std::ffi::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__convchart_interfaces__msg__InferenceResult() }
  }
}


