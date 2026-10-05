std_msgs/Header header

geometry_msgs/PoseWithCovariance pose
bool covariance_valid #if false, then fallback covariance is used
float32 rms #RMS of pose result
uint8 num_used #num of corners 
string reason #if non-empty, then the inference result is invalid
bool ambiguous #if ambiguous, compute suprise


geometry_msgs/PoseWithCovariance pose_alt
bool covariance_valid_alt #if false, then fallback covariance is used
float32 rms_alt #RMS of pose result
uint8 num_used_alt #num of corners 
