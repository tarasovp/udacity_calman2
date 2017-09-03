#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

#define EPS 0.001
#define SKIP_RADAR 0
#define SKIP_LIDAR 0

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
    // if this is false, laser measurements will be ignored (except during init)
    use_laser_ = true;
    
    // if this is false, radar measurements will be ignored (except during init)
    use_radar_ = true;
    
    // [pos1 pos2 vel_abs yaw_angle yaw_rate]
    n_x_ = 5;
    
    // initial state vector
    x_ = VectorXd(n_x_);
    
    //Augmented state dimensions : x dimensions + 2 (nu and psi)
    n_aug_ = n_x_ + 2;
    
    //initial augmented state vector
    x_aug_ = VectorXd(n_aug_);
    
    // initial covariance matrix
    P_ = MatrixXd(n_x_, n_x_);
    
    // aug state covariance matrix
    P_aug_ = MatrixXd(n_aug_, n_aug_);
    
    
    // Sigma Points
    Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
    
    time_us_ = 0.0f;
    // Process noise standard deviation longitudinal acceleration in m/s^2
    std_a_ = 1.5f;
    
    // Process noise standard deviation yaw acceleration in rad/s^2
    std_yawdd_ = 0.8f;
    
    // Laser measurement noise standard deviation position1 in m
    std_laspx_ = 0.15f;
    
    // Laser measurement noise standard deviation position2 in m
    std_laspy_ = 0.15f;
    
    // Radar measurement noise standard deviation radius in m
    std_radr_ = 0.3f;
    
    // Radar measurement noise standard deviation angle in rad
    std_radphi_ = 0.03f;
    
    // Radar measurement noise standard deviation radius change in m/s
    std_radrd_ = 0.3f;
    
    // Weights of sigma points
    weights_ = VectorXd(2*n_aug_+1);
    
    // Sigma point spreading parameter
    lambda_ = 3 - n_aug_;
    
    R_radar= MatrixXd(3,3);
    R_radar<<    std_radr_*std_radr_, 0, 0,
    0, std_radphi_*std_radphi_, 0,
    0, 0, std_radrd_*std_radrd_;
    
    R_lidar= MatrixXd(2,2);
    R_lidar<<      std_laspx_*std_laspx_, 0,
    0, std_laspy_*std_laspy_;
    
    NIS_radar_ = 0;
    NIS_lidar_ = 0;
    
    n_sig_ = 2 * n_aug_ +1;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */


void UKF::GenerateSigmaPoints(MatrixXd* Xsig_out) {
    
    double lambda = 3 - n_x_;
    //set state dimension
     //set example state
     //create sigma point matrix
    MatrixXd Xsig = MatrixXd(n_x_, 2 * n_x_ + 1);
    
    //calculate square root of P
    MatrixXd A = P_.llt().matrixL();
    
    //set first column of sigma point matrix
    Xsig.col(0)  = x_;
    
    //set remaining sigma points
    for (int i = 0; i < n_x_; i++)
    {
        Xsig.col(i+1)     = x_ + sqrt(lambda+n_x_) * A.col(i);
        Xsig.col(i+1+n_x_) = x_ - sqrt(lambda+n_x_) * A.col(i);
    }
    
     *Xsig_out = Xsig;
}

void UKF::AugmentedSigmaPoints(MatrixXd* Xsig_out) {
     //create sigma point matrix
     MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
     
     //create augmented mean state
    x_aug_.head(5) = x_;
    x_aug_(5) = 0;
    x_aug_(6) = 0;
    
    //create augmented covariance matrix
    P_aug_.fill(0.0);
    P_aug_.topLeftCorner(5,5) = P_;
    P_aug_(5,5) = std_a_*std_a_;
    P_aug_(6,6) = std_yawdd_*std_yawdd_;
    
    //create square root matrix
    MatrixXd L = P_aug_.llt().matrixL();
    
    //create augmented sigma points
    Xsig_aug.col(0)  = x_aug_;
    for (int i = 0; i< n_aug_; i++)
    {
        Xsig_aug.col(i+1)       = x_aug_ + sqrt(lambda_+n_aug_) * L.col(i);
        Xsig_aug.col(i+1+n_aug_) = x_aug_ - sqrt(lambda_+n_aug_) * L.col(i);
    }
    
    //write result
    *Xsig_out = Xsig_aug;
    
    
}


void UKF::SigmaPointPrediction(const MatrixXd Xsig_aug, const double delta_t) {
    
    
    //create matrix with predicted sigma points as columns
    
    //predict sigma points
    for (int i = 0; i< 2*n_aug_+1; i++)
    {
        //extract values for better readability
        double p_x = Xsig_aug(0,i);
        double p_y = Xsig_aug(1,i);
        double v = Xsig_aug(2,i);
        double yaw = Xsig_aug(3,i);
        double yawd = Xsig_aug(4,i);
        double nu_a = Xsig_aug(5,i);
        double nu_yawdd = Xsig_aug(6,i);
        
        //predicted state values
        double px_p, py_p;
        
        //avoid division by zero
        if (fabs(yawd) > 0.001) {
            px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
            py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
        }
        else {
            px_p = p_x + v*delta_t*cos(yaw);
            py_p = p_y + v*delta_t*sin(yaw);
        }
        
        double v_p = v;
        double yaw_p = yaw + yawd*delta_t;
        double yawd_p = yawd;
        
        //add noise
        px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
        py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
        v_p = v_p + nu_a*delta_t;
        
        yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
        yawd_p = yawd_p + nu_yawdd*delta_t;
        
        //write predicted sigma point into right column
        Xsig_pred_(0,i) = px_p;
        Xsig_pred_(1,i) = py_p;
        Xsig_pred_(2,i) = v_p;
        Xsig_pred_(3,i) = yaw_p;
        Xsig_pred_(4,i) = yawd_p;
    }
    
}

void UKF::PredictMeanAndCovariance() {
    
    
    
    // set weights
    weights_.fill(0.5 / (lambda_ + n_aug_));
    weights_(0) = lambda_/(lambda_ + n_aug_);
    
    //predicted state mean
    x_ = Xsig_pred_ * weights_;
    
    //predicted state covariance matrix
    P_.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
        
        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        //angle normalization
        while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
        while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;
        
        P_ = P_ + weights_(i) * x_diff * x_diff.transpose() ;
    }
    
    
    
}


void UKF::ProcessMeasurement(MeasurementPackage measurement_pack) {
    if (!is_initialized_) {
        /**
         TODO:
         * Initialize the state ekf_.x_ with the first measurement.
         * Create the covariance matrix.
         * Remember: you'll need to convert radar from polar to cartesian coordinates.
         */
        x_ << 0, 0, 0, 0, 0;
        
        P_ << 0.15,    0, 0, 0, 0,
        0, 0.15, 0, 0, 0,
        0,    0, 1, 0, 0,
        0,    0, 0, 1, 0,
        0,    0, 0, 0, 1;
        
        // first measurement
        if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
            /**
             Convert radar from polar to cartesian coordinates and initialize state.
             */
            float rho = measurement_pack.raw_measurements_[0]; // range
            float phi = measurement_pack.raw_measurements_[1]; // bearing
            float rho_dot = measurement_pack.raw_measurements_[2]; // velocity of rho
            // Coordinates convertion from polar to cartesian
            float x = rho * cos(phi);
            float y = rho * sin(phi);
            x_ << x, y, rho_dot, 0, 0;
        }
        else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
            // We don't know velocities from the first measurement of the LIDAR, so, we use zeros
            x_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0, 0, 0;
        }
        // Deal with the special case initialisation problems
        if (fabs(x_(0)) < EPS and fabs(x_(1)) < EPS) {
            x_(0) = EPS;
            x_(1) = EPS;
        }
        
        //init for augment X
        x_aug_.head(5) = x_;
        x_aug_(5) = 0;
        x_aug_(6) = 0;
        
        // Initial covariance matrix
        
        // Print the initialization results
        //cout << "EKF init: " << ekf_.x_ << endl;
        // Save the initiall timestamp for dt calculation
        previous_timestamp_ = measurement_pack.timestamp_;
        // Done initializing, no need to predict or update
        is_initialized_ = true;
        return;
    }
    
    // Compute the time elapsed between the current and previous measurements
    float dt = (measurement_pack.timestamp_ - previous_timestamp_);
    dt /= 1000000.0; // convert micros to s
    previous_timestamp_ = measurement_pack.timestamp_;
    
    Prediction(dt);
    
    
    //Update
    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
        if (!SKIP_RADAR) {
            UpdateRadar(measurement_pack);
        }
    } else {
        if (!SKIP_LIDAR) {
            UpdateLidar(measurement_pack);
        }
    }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
    MatrixXd Xsig = MatrixXd(n_x_, 2 * n_x_ + 1);
    MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
    
    GenerateSigmaPoints(&Xsig);
    AugmentedSigmaPoints(&Xsig_aug);
   
    SigmaPointPrediction(Xsig_aug, delta_t);
   
    PredictMeanAndCovariance();
    
   }

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
        NIS_lidar_=UpdateState(  meas_package);
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
    
    NIS_radar_=UpdateState(  meas_package);
}

float UKF::UpdateState(MeasurementPackage meas_package) {
    
    int n_z = meas_package.sensor_type_ == MeasurementPackage::RADAR ? 3:2;
    MatrixXd R = meas_package.sensor_type_ == MeasurementPackage::RADAR ? R_radar:R_lidar;

    
    MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
    MatrixXd S = MatrixXd(n_z,n_z);
    VectorXd z_pred = VectorXd(n_z);
    
    S.fill(0.0);
    Zsig.fill(0.0);
    z_pred.fill(0.0);
    
    //transform sigma points into measurement space
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
        
        if (n_z==3)
        {
            double px = Xsig_pred_(0,i);
            double py = Xsig_pred_(1,i);
            double v  = Xsig_pred_(2,i);
            double yaw = Xsig_pred_(3,i);
            double v1 = cos(yaw)*v;
            double v2 = sin(yaw)*v;
            
            //check for zeros
            if (fabs(px) < EPS) {
                px = EPS;
            }
            if (fabs(py) < EPS) {
                py = EPS;
            }
            
            Zsig(0,i) = sqrt(px*px + py*py);
            Zsig(1,i) = atan2(py,px);
            Zsig(2,i) = (px*v1 + py*v2 ) / sqrt(px*px + py*py);
        }
        else
        {
            double px = Xsig_pred_(0,i);
            double py = Xsig_pred_(1,i);
            
            //sigma point predictions in measurement space
            Zsig(0,i) = px;
            Zsig(1,i) = py;

        }
    }
    
    //mean predicted measurement
    z_pred = Zsig * weights_; //Matrix Calc (3X15) * (15X1) = (3X1)
    
    //measurement covariance matrix S
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;
        
        //angle normalization
        if (n_z==3){
            while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
            while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;
        }
        
        S = S + weights_(i) * z_diff * z_diff.transpose();
    }
    
    //add measurement noise covariance matrix
    
    S = S + R;
    
    //create matrix for cross correlation Tc
    MatrixXd Tc = MatrixXd(n_x_, n_z);
    
    
    
    //calculate cross correlation matrix
    Tc.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
        
        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;
        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        
        //angle normalization
        if (n_z==3){
            while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
            while (z_diff(1)<=-M_PI) z_diff(1)+=2.*M_PI;
            while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
            while (x_diff(3)<=-M_PI) x_diff(3)+=2.*M_PI;
            
        }
        
        
        Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
    }
    
    //Kalman gain K;
    MatrixXd K = Tc * S.inverse();
    
    //residual
    VectorXd z_diff = meas_package.raw_measurements_ - z_pred;
    
    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<=-M_PI) z_diff(1)+=2.*M_PI;
    
    //update state mean and covariance matrix
    x_ = x_ + K * z_diff;
    P_ = P_ - K*S*K.transpose();
    //NIS Update
    NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;
 
}
