#include "ukf.h"
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::pow;


// Given a yaw angle in rad, returns the equivalent in the range -pi to pi
double bounded_angle(double angle) {
  angle = std::fmod(angle + M_PI, 2 * M_PI);  // angle in rad
  if (angle < 0) angle += 2 * M_PI;
  return angle - M_PI;
}

// Construct the unscented Kalman filter (UKF). 
// Constant turn-rate and velocity magnitude (CTRV) motion model is used.

UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd::Identity(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  
  std_a_ = 2.0;        // can be tuned

  // Process noise standard deviation yaw acceleration in rad/s^2
  
  std_yawdd_ = 0.5;    // can be tuned
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values
   */
  
  n_x_ = 5;
  n_aug_ = 7;
  n_sig_ = 2 * n_aug_ + 1;
  weights_ = VectorXd(n_sig_);
  lambda_ = 3.0 - n_x_;

  Xsig_pred_ = MatrixXd(n_x_, n_sig_);
  Xsig_pred_.fill(0.0);

  // Setup augmented weights vector
  double w0 = lambda_ / (lambda_ + n_aug_);     // first weight element 
  double w = 1 / (2 * (lambda_ + n_aug_));      // weight elements from 2 to n_aug
  weights_.fill(w);
  weights_(0) = w0;

  is_initialized_ = false;
}


UKF::~UKF() {}


// Initialize each UKF instance exactly once
void UKF::InitializeUKF(MeasurementPackage meas_package) {

  // Use the first measurement to set the value of the mean state
  x_.fill(0.0);
  x_.head(2) << meas_package.raw_measurements_;

  time_us_ = meas_package.timestamp_;
  is_initialized_ = true;
}

// UKF prediction step, estimate the object's location at next timestep k+1
void UKF::Prediction(double delta_t) {
  
  // STEP I: Generate sigma points (sampling) in the augmented state space
  // cout << "Generating sigma points in the augmented state space" << endl;
  
  MatrixXd Xsig = MatrixXd(n_aug_, n_sig_);
  Xsig.fill(0.0);
  VectorXd x_aug = VectorXd(n_aug_);
  x_aug.fill(0.0);
  x_aug.head(n_x_) = x_;                      // Augment the mean state

  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);  // Augment the covariance matrix
  P_aug.fill(0.0);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(n_x_, n_x_) = pow(std_a_, 2);
  P_aug(n_x_ + 1, n_x_ + 1) = pow(std_yawdd_, 2);

  MatrixXd A_aug = P_aug.llt().matrixL();     // Create the square root matrix A
  double c_aug = sqrt(lambda_ + n_aug_);
  MatrixXd cA_aug = c_aug * A_aug;
  
  Xsig.col(0) = x_aug;                        // First column of Xsig is central sigma point
  
  for (int i = 1; i <= n_aug_; i++) {
    Xsig.col(i) = x_aug + cA_aug.col(i - 1);              // First group of sigma points
  }
  for (int i = n_aug_ + 1; i <= n_sig_ - 1; i++) {
    Xsig.col(i) = x_aug - cA_aug.col(i - 1 - n_aug_);     // Symmetric group
  }

  // STEP II: Predict the motion of each sigma point
  // cout << "Predicting the motion of each sigma point" << endl; 

  double dt = delta_t;
  Xsig_pred_.fill(0.0);

  // Loop over each sigma point, transforming back to the original state space
  for (int i = 0; i < n_sig_; i++) {
    double px = Xsig(0, i);
    double py = Xsig(1, i);
    double v = Xsig(2, i); 
    double yaw = Xsig(3, i);
    double yawd = Xsig(4, i);
    double nu_a = Xsig(5, i);
    double nu_yawdd = Xsig(6, i);

    // Transform from the augmented to the original (n_x-dim) state space
    
    // Avoid division by zero
    if (std::fabs(yawd) > 0.001) {
      Xsig_pred_(0, i) = px + v / yawd * (sin(yaw + yawd * dt) - sin(yaw)) +
                        pow(dt, 2) / 2 * cos(yaw) * nu_a; 
      Xsig_pred_(1, i) = py + v / yawd * (-cos(yaw + yawd * dt) + cos(yaw)) +
                        pow(dt, 2) / 2 * sin(yaw) * nu_a;
    } else {
      Xsig_pred_(0, i) = px + v * dt * cos(yaw) +
                         pow(dt, 2) / 2 * cos(yaw) * nu_a;
      Xsig_pred_(1, i) = py + v * dt * sin(yaw) +
                         pow(dt, 2) / 2 * sin(yaw) * nu_a;
    }
    
    Xsig_pred_(2, i) = v + 0 + dt * nu_a;
    Xsig_pred_(3, i) = yaw + yawd * dt + pow(dt, 2) / 2 * nu_yawdd;
    Xsig_pred_(4, i) = yawd + 0 + dt * nu_yawdd;
  }

  // STEP III: Predict the next state: (mean and covariance matrices)
  // cout << "Predicting the next state's mean and covariance" << endl;
  
  // Predict the mean state
  VectorXd x = VectorXd(n_x_);
  x.fill(0.0);  
  for (int i = 0; i < n_sig_; i++) {
    x = x + weights_(i) * Xsig_pred_.col(i);
  }
  
  // Predict the covariance matrix
  MatrixXd P = MatrixXd(n_x_, n_x_);
  P.fill(0.0);  
  for (int i = 0; i < n_sig_; i++) {
    P = P + weights_(i) * (Xsig_pred_.col(i) - x_) * (Xsig_pred_.col(i) - x_).transpose();
  }

  x_ = x;
  P_ = P; 
}


// Branch to a lidar or radar measurement update, depending on sensor type
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  if (!is_initialized_) {
    // Skip the prediction step for the initial measurement,
    InitializeUKF(meas_package);
    
    if (MeasurementPackage::SensorType::LASER == meas_package.sensor_type_) {
      UpdateLidar(meas_package);
    }
    else if (MeasurementPackage::SensorType::RADAR == meas_package.sensor_type_) {
      UpdateRadar(meas_package);
    }
  }
  else {
    // On each subsequent measurement, trigger a full prediction/update cycle
    double delta_t = (meas_package.timestamp_ - time_us_) / 1e6;
    time_us_ = meas_package.timestamp_;
    Prediction(delta_t);

    if (MeasurementPackage::SensorType::LASER == meas_package.sensor_type_) {
      UpdateLidar(meas_package);
    }
    else if (MeasurementPackage::SensorType::RADAR == meas_package.sensor_type_) {
      UpdateRadar(meas_package);
    }
  }
}

// Use lidar data to update the belief about the object's current position
void UKF::UpdateLidar(MeasurementPackage meas_package) {

  // STEP IV: Predict the measurement mean and covariance; calculate Kalman gain
  // cout << "Predicting the measurement mean and covariance; calculating Kalman gain" << endl; 
  
  VectorXd z = meas_package.raw_measurements_;
  int n_z = z.size();                             // Measurement z is a 2x1 vector for lidar
  
  // Measurement matrix
  MatrixXd H = MatrixXd(n_z, n_x_);
  H << 1, 0, 0, 0, 0,
       0, 1, 0, 0, 0;

  // Measurement covariance matrix
  MatrixXd R = MatrixXd(n_z, n_z);
  R << pow(std_laspx_, 2), 0,
       0, pow(std_laspy_, 2);

  VectorXd z_pred = VectorXd(n_z);
  z_pred = x_.head(n_z);                        // Extract the px, py values from the state

  VectorXd y = z - z_pred;                      // Calculate the residuals vector y
  MatrixXd Ht = H.transpose();
  MatrixXd S = H * P_ * Ht + R;
  MatrixXd Sinv = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Sinv;

  // STEP V: Update the state, by applying the Kalman gain to the residual
  // cout << "Updating the state by applying the Kalman gain to the residual" << endl; 

  MatrixXd I = MatrixXd::Identity(n_x_, n_x_);

  // Update the mean and covariance matrix
  x_ = x_ + (K * y);
  P_ = (I - K * H) * P_;

  // Calculate normalized innovation squared (NIS) for tuning (Commented)
  // double NIS = y.transpose() * Sinv * y;
  // cout << "Lidar NIS (2-df X^2, 95% < 5.991) = " << NIS << endl; 
}

// Use radar data to update the belief about the object's current position
void UKF::UpdateRadar(MeasurementPackage meas_package) {

  // STEP 4) Predict the measurement mean (z_pred) and innovation covariance (S)
  // cout << "Predicting the measurement mean and covariance; calculating Kalman gain" << endl; 
  
  VectorXd z = meas_package.raw_measurements_;
  int n_z = z.size();               // Measurement z is a 3x1 vector for radar

  MatrixXd Zsig = MatrixXd(n_z, n_sig_);
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);

  // Transform the sigma points into measurement space
  for (int i = 0; i < n_sig_; i++) {
      double px = Xsig_pred_(0, i);
      double py = Xsig_pred_(1, i);
      double v = Xsig_pred_(2, i);
      double yaw = Xsig_pred_(3, i);
      double yawd = Xsig_pred_(4, i);
      
      double rho = sqrt(pow(px, 2) + pow(py, 2));
      double phi = std::atan2(py, px);
      double rhod = 0.0;
      if (std::fabs(rho) > 0.001) {
        rhod = (px * cos(yaw) * v + py * sin(yaw) * v) / rho; 
      }

      Zsig(0, i) = rho;
      Zsig(1, i) = phi;
      Zsig(2, i) = rhod;
  }

  // Calculate the predicted mean measurement z_pred
  for (int i = 0; i < n_sig_; i++) {
    z_pred = z_pred + weights_(i) * Zsig.col(i);  
  }
  
  // Calculate the innovation covariance matrix S
  MatrixXd R = MatrixXd(n_z, n_z);
  MatrixXd S = MatrixXd(n_z, n_z);
  S.fill(0.0);

  for (int i = 0; i < n_sig_; i++) {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    z_diff(1) = bounded_angle(z_diff(1));

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  double std_rho2 = pow(std_radr_, 2);
  double std_phi2 = pow(std_radphi_, 2);
  double std_rhod2 = pow(std_radrd_, 2);
  double mod_angle = 0.0;

  R << std_rho2, 0, 0,
       0, std_phi2, 0,
       0, 0, std_rhod2;

  S = S + R;

  // STEP 5) Update the state, by applying the Kalman gain to the residual
  // cout << "Updating the state by applying the Kalman gain to the residual" << endl; 

  // Calculate the cross correlation matrix Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0.0);

  for (int i = 0; i < n_sig_; i++) {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    z_diff(1) = bounded_angle(z_diff(1));

    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    x_diff(3) = bounded_angle(x_diff(3));

    Tc += weights_(i) * x_diff * z_diff.transpose();
  }

  // Calculate the Kalman gain
  MatrixXd K = MatrixXd(n_x_, n_z); 
  MatrixXd Sinv = S.inverse();
  K = Tc * Sinv;

  VectorXd residuals = z - z_pred;
  residuals(1) = bounded_angle(residuals(1));

  // Update the mean and covariance matrix
  x_ = x_ + K * residuals;
  MatrixXd Kt = K.transpose();
  P_ = P_ - K * S * Kt;

  // Calculate normalized innovation squared (NIS) for tuning (Commented)
  // double NIS = residuals.transpose() * Sinv * residuals;
  // cout << "Radar NIS (3-df X^2, 95% < 7.815) = " << NIS << endl;

}
