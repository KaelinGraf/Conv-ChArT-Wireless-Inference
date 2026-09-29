#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};



// Corresponds to convchart_interfaces__msg__InferenceResult

// This struct is not documented.
#[allow(missing_docs)]

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct InferenceResult {

    // This member is not documented.
    #[allow(missing_docs)]
    pub header: std_msgs::msg::Header,


    // This member is not documented.
    #[allow(missing_docs)]
    pub pose: geometry_msgs::msg::PoseWithCovariance,

    /// RMS of pose result
    pub rms: f32,

    /// num of corners
    pub num_used: u8,

    /// if non-empty, then the inference result is invalid
    pub reason: std::string::String,

    /// if ambiguous, compute suprise
    pub ambiguous: bool,


    // This member is not documented.
    #[allow(missing_docs)]
    pub pose_alt: geometry_msgs::msg::PoseWithCovariance,

    /// RMS of pose result
    pub rms_alt: f32,

    /// num of corners
    pub num_used_alt: u8,

}



impl Default for InferenceResult {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::msg::rmw::InferenceResult::default())
  }
}

impl rosidl_runtime_rs::Message for InferenceResult {
  type RmwMsg = super::msg::rmw::InferenceResult;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Owned(msg.header)).into_owned(),
        pose: geometry_msgs::msg::PoseWithCovariance::into_rmw_message(std::borrow::Cow::Owned(msg.pose)).into_owned(),
        rms: msg.rms,
        num_used: msg.num_used,
        reason: msg.reason.as_str().into(),
        ambiguous: msg.ambiguous,
        pose_alt: geometry_msgs::msg::PoseWithCovariance::into_rmw_message(std::borrow::Cow::Owned(msg.pose_alt)).into_owned(),
        rms_alt: msg.rms_alt,
        num_used_alt: msg.num_used_alt,
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Borrowed(&msg.header)).into_owned(),
        pose: geometry_msgs::msg::PoseWithCovariance::into_rmw_message(std::borrow::Cow::Borrowed(&msg.pose)).into_owned(),
      rms: msg.rms,
      num_used: msg.num_used,
        reason: msg.reason.as_str().into(),
      ambiguous: msg.ambiguous,
        pose_alt: geometry_msgs::msg::PoseWithCovariance::into_rmw_message(std::borrow::Cow::Borrowed(&msg.pose_alt)).into_owned(),
      rms_alt: msg.rms_alt,
      num_used_alt: msg.num_used_alt,
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      header: std_msgs::msg::Header::from_rmw_message(msg.header),
      pose: geometry_msgs::msg::PoseWithCovariance::from_rmw_message(msg.pose),
      rms: msg.rms,
      num_used: msg.num_used,
      reason: msg.reason.to_string(),
      ambiguous: msg.ambiguous,
      pose_alt: geometry_msgs::msg::PoseWithCovariance::from_rmw_message(msg.pose_alt),
      rms_alt: msg.rms_alt,
      num_used_alt: msg.num_used_alt,
    }
  }
}


