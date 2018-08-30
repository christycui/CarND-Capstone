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
        ctrans_tosearch = self.convert_color(img_tosearch, conv='BGR2YCrCb')
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
                hog_feat1 = self.get_hog_features(subimg[:,:,0], orient, pix_per_cell, cell_per_block, feature_vec=True).ravel() 
                hog_feat2 = self.get_hog_features(subimg[:,:,1], orient, pix_per_cell, cell_per_block, feature_vec=True).ravel()
                hog_feat3 = self.get_hog_features(subimg[:,:,2], orient, pix_per_cell, cell_per_block, feature_vec=True).ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
                
                # Get color features
                spatial_features = self.bin_spatial(subimg, size=spatial_size)
                hist_features = self.color_hist(subimg, nbins=hist_bins)
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

	def convert_color(self, img, conv='RGB2YCrCb'):
		if conv == 'RGB2YCrCb':
			return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    		if conv == 'BGR2YCrCb':
    	    		return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    		if conv == 'BGR2LUV':
    	    		return cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
    		if conv == 'BGR2HSV':
    	    		return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    		if conv == 'BGR2HLS':
    	    		return cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    		if conv == 'BGR2YUV':
    	    		return cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    		if conv == 'BGR2RGB':
    	    		return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	def bin_spatial(self, img, size=(32, 32)):
    		# Use cv2.resize().ravel() to create the feature vector
    		features = cv2.resize(img, size).ravel() 
    		# Return the feature vector
    		return features

# Define a function to compute color histogram features  
	def color_hist(self, img, nbins=32, bins_range=(0, 256)):
    		# Compute the histogram of the color channels separately
    		channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    		channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    		channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    		# Concatenate the histograms into a single feature vector
    		hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    		# Return the individual histograms, bin_centers and feature vector
    		return hist_features

	def get_hog_features(self, img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    	# Call with two outputs if vis==True
    		if vis == True:
    	    		features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
    	                              cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
    	                              visualise=vis, feature_vector=feature_vec)
    	    		return features, hog_image
    	# Otherwise call with one output
    		else:   
    	    		features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
    	                   cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
    	                   visualise=vis, feature_vector=feature_vec)
    	    		return features
