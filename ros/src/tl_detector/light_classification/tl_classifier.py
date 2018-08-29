from styx_msgs.msg import TrafficLight
import pickle
import rospy

class TLClassifier(object):
    def __init__(self):
        self.x = 109 # detection image size
        self.y = 43
	self.model_path = rospy.get_param('/model_path')
        # load pickle file
        self.X_scaler = pickle.load(open(self.model_path+"X_scaler.pkl", "rb"))
        self.svc = pickle.load(open(self.model_path+'svc.pkl', 'rb'))

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        orient = 7  # HOG orientations
        pix_per_cell = 16 # HOG pixels per cell
        cell_per_block = 2 # HOG cells per block
        spatial_size = (16, 16) # Spatial binning dimensions
        hist_bins = 16    # Number of histogram bins

        ystart = 0
        ystop = 600
        scale = 2
        result = self.find_red_light(image, ystart, ystop, scale, self.svc, \
            self.X_scaler, orient, pix_per_cell, cell_per_block, \
            spatial_size, hist_bins)
        return TrafficLight.RED if result else TrafficLight.UNKNOWN

    def find_red_light(self, img, ystart, ystop, scale, svc, X_scaler, orient, \
        pix_per_cell, cell_per_block, spatial_size, hist_bins):
        img_tosearch = img[ystart:ystop,:,:]
        ctrans_tosearch = convert_color(img_tosearch, conv='BGR2YCrCb')
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
            
        ch1 = ctrans_tosearch[:,:,0]
        ch2 = ctrans_tosearch[:,:,1]
        ch3 = ctrans_tosearch[:,:,2]

        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // pix_per_cell)-1
        nyblocks = (ch1.shape[0] // pix_per_cell)-1 
        nfeat_per_block = orient*cell_per_block**2
        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // pix_per_cell)-1 
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step
        
        # Compute individual channel HOG features for the entire image
        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step
                xleft = xpos*pix_per_cell
                ytop = ypos*pix_per_cell
                
                if ytop+x < 600 or xleft+y < 800:
                    pass
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop+self.x, xleft:xleft+self.y], (x,y))
                
                # HOG features
                hog_feat1 = get_hog_features(subimg[:,:,0], orient, pix_per_cell, cell_per_block, feature_vec=True).ravel() 
                hog_feat2 = get_hog_features(subimg[:,:,1], orient, pix_per_cell, cell_per_block, feature_vec=True).ravel()
                hog_feat3 = get_hog_features(subimg[:,:,2], orient, pix_per_cell, cell_per_block, feature_vec=True).ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
                
                # Get color features
                spatial_features = bin_spatial(subimg, size=spatial_size)
                hist_features = color_hist(subimg, nbins=hist_bins)
                # Scale features and make a prediction
                test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
                test_prediction = svc.predict(test_features)
                
                if test_prediction == 1:
                    xbox_left = np.int(xleft*scale)
                    ytop_draw = np.int(ytop*scale)
                    win_draw = np.int(window*scale)
                    boxes.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
                    return 1
        return 0
