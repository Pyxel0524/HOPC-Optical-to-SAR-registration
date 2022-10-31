import math

import numpy as np
import cv2 # Faster Fourier transforms than NumPy and Scikit-Image
from PCV.localdescriptors import harris
import matplotlib.pyplot as plt



#####HOPC点匹配
def block_Harris(img):
    harrisim = harris.compute_harris_response(img, 3)
    harrisim = np.nan_to_num(harrisim)
    filtered_coords = harris.get_harris_points(harrisim,8,threshold = 0.1)
    return filtered_coords

def PhaseCongruency(InputImage, NumberScales, NumberAngles):

    # nscale           4    - Number of wavelet scales, try values 3-6
    # norient          6    - Number of filter orientations.
    # minWaveLength    3    - Wavelength of smallest scale filter.
    # mult             2.1  - Scaling factor between successive filters.
    # sigmaOnf         0.55 - Ratio of the standard deviation of the Gaussian
    #                         describing the log Gabor filter's transfer function
    #                         in the frequency domain to the filter center frequency.
    # k                2.0  - No of standard deviations of the noise energy beyond
    #                         the mean at which we set the noise threshold point.
    #                         You may want to vary this up to a value of 10 or
    #                         20 for noisy images
    # cutOff           0.5  - The fractional measure of frequency spread
    #                         below which phase congruency values get penalized.
    # g                10   - Controls the sharpness of the transition in
    #                         the sigmoid function used to weight phase
    #                         congruency for frequency spread.
    # noiseMethod      -1   - Parameter specifies method used to determine
    #                         noise statistics.
    #                           -1 use median of smallest scale filter responses
    #                           -2 use mode of smallest scale filter responses
    #                            0+ use noiseMethod value as the fixed noise threshold
    minWaveLength = 3
    mult = 2.1
    sigmaOnf = 0.55
    k = 2.0
    cutOff = 0.5
    g = 10
    noiseMethod = -1


    epsilon = .0001 # Used to prevent division by zero.


    f_cv = cv2.dft(np.float32(InputImage),flags=cv2.DFT_COMPLEX_OUTPUT)

    #------------------------------
    nrows, ncols = InputImage.shape
    zero = np.zeros((nrows,ncols))
    EO = np.zeros((nrows,ncols,NumberScales,NumberAngles),dtype=complex)
    PC = np.zeros((nrows,ncols,NumberAngles))
    covx2 = np.zeros((nrows,ncols))
    covy2 = np.zeros((nrows,ncols))
    covxy = np.zeros((nrows,ncols))
    EnergyV = np.zeros((nrows,ncols,3))
    pcSum = np.zeros((nrows,ncols))

    #分配方向
    a = np.zeros((nrows,ncols))
    b = np.zeros((nrows,ncols))

    # Matrix of radii
    cy = math.floor(nrows/2)
    cx = math.floor(ncols/2)
    y, x = np.mgrid[0:nrows, 0:ncols]
    y = (y-cy)/nrows
    x = (x-cx)/ncols

    radius = np.sqrt(x**2 + y**2)
    radius[cy, cx] = 1

    # Matrix values contain polar angle.
    # (note -ve y is used to give +ve anti-clockwise angles)
    theta = np.arctan2(-y, x)
    sintheta = np.sin(theta)
    costheta = np.cos(theta)

    # Initialise set of annular bandpass filters
    #  Here I use the method of scale selection from the code I used to generate
    #   stimuli for my latest experiments (spatial feature scaling):
    #   /Users/carl/Studies/Face_Projects/features_wavelet
    #NumberScales = 3 # should be odd
    annularBandpassFilters = np.empty((nrows,ncols,NumberScales))


    # Number of filter orientations.
    #NumberAngles = 6
    """ Ratio of angular interval between filter orientations and the standard deviation
        of the angular Gaussian function used to construct filters in the freq. plane.
    """

    # The following implements the log-gabor transfer function
    """ From http://www.peterkovesi.com/matlabfns/PhaseCongruency/Docs/convexpl.html
        The filter bandwidth is set by specifying the ratio of the standard deviation
        of the Gaussian describing the log Gabor filter's transfer function in the
        log-frequency domain to the filter center frequency. This is set by the parameter
        sigmaOnf . The smaller sigmaOnf is the larger the bandwidth of the filter.
        I have not worked out an expression relating sigmaOnf to bandwidth, but
        empirically a sigmaOnf value of 0.75 will result in a filter with a bandwidth
        of approximately 1 octave and a value of 0.55 will result in a bandwidth of
        roughly 2 octaves.
    """
    # sigmaOnf = 0.74  # approximately 1 octave
    # sigmaOnf = 0.55  # approximately 2 octaves
    """ From Wilson, Loffler and Wilkinson (2002 Vision Research):
        The bandpass filtering alluded to above was used because of ubiquitous evidence
        that face discrimination is optimal within a 2.0 octave (at half amplitude)
        bandwidth centered upon 8–13 cycles per face width (Costen et al., 1996;
        Fiorentini et al., 1983; Gold et al., 1999; Hayes et al., 1986; Näsänen, 1999).
        We therefore chose a radially symmetric filter with a peak frequency of 10.0
        cycles per mean face width and a 2.0 octave bandwidth described by a difference
         of Gaussians (DOG):"""

    # Lowpass filter to remove high frequency 'garbage'
    filterorder = 15  # filter 'sharpness'
    cutoff = .45
    normradius = radius / (abs(x).max()*2)
    lowpassbutterworth = 1.0 / (1.0 + (normradius / cutoff)**(2*filterorder))
    #
    # Note: lowpassbutterworth is currently DC centered.

    for s in np.arange(NumberScales):
        wavelength = minWaveLength*mult**s
        fo = 1.0/wavelength                  # Centre frequency of filter.
        logGabor = np.exp((-(np.log(radius/fo))**2) / (2 * math.log(sigmaOnf)**2))
        annularBandpassFilters[:,:,s] = logGabor*lowpassbutterworth  # Apply low-pass filter
        annularBandpassFilters[cy,cx,s] = 0          # Set the value at the 0 frequency point of the filter
                                                     # back to zero (undo the radius fudge).

    # main loop
    for o in np.arange(NumberAngles):
        # Construct the angular filter spread function
        angl = o*math.pi/NumberAngles # Filter angle.
        # For each point in the filter matrix calculate the angular distance from
        # the specified filter orientation.  To overcome the angular wrap-around
        # problem sine difference and cosine difference values are first computed
        # and then the atan2 function is used to determine angular distance.

        # % Scale theta so that cosine spread function has the right wavelength and clamp to pi
        # dtheta = min(dtheta*norient/2,pi);
        # % The spread function is cos(dtheta) between -pi and pi.  We add 1,
        # % and then divide by 2 so that the value ranges 0-1
        """ For each point in the filter matrix calculate the angular distance from the
            specified filter orientation.  To overcome the angular wrap-around problem
            sine difference and cosine difference values are first computed and then
            the atan2 function is used to determine angular distance.
        """
        ds = sintheta * math.cos(angl) - costheta * math.sin(angl)      # Difference in sine.
        dc = costheta * math.cos(angl) + sintheta * math.sin(angl)      # Difference in cosine.
        dtheta = np.abs(np.arctan2(ds,dc))                              # Absolute angular distance.

        # Scale theta so that cosine spread function has the right wavelength
        #   and clamp to pi
        dtheta = np.minimum(dtheta*NumberAngles/2, math.pi)

        #spread = np.exp((-dtheta**2) / (2 * thetaSigma**2));  # Calculate the angular
                                                              # filter component.
        # The spread function is cos(dtheta) between -pi and pi.  We add 1,
        #   and then divide by 2 so that the value ranges 0-1
        spread = (np.cos(dtheta)+1)/2

        sumE_ThisOrient   = np.zeros((nrows,ncols))  # Initialize accumulator matrices.
        sumO_ThisOrient   = np.zeros((nrows,ncols))
        sumAn_ThisOrient  = np.zeros((nrows,ncols))
        Energy            = np.zeros((nrows,ncols))

        maxAn = []
        for s in np.arange(NumberScales):
            filter = annularBandpassFilters[:,:,s] * spread # Multiply radial and angular
                                                            # components to get the filter.

            criticalfiltershift = np.fft.ifftshift( filter )
            criticalfiltershift_cv = np.empty((nrows, ncols, 2))
            for ip in range(2):
                criticalfiltershift_cv[:,:,ip] = criticalfiltershift

            # Convolve image with even and odd filters returning the result in EO
            MatrixEO = cv2.idft( criticalfiltershift_cv * f_cv )
            EO[:,:,s,o] = MatrixEO[:,:,1] + 1j*MatrixEO[:,:,0]

            An = cv2.magnitude(MatrixEO[:,:,0], MatrixEO[:,:,1])    # Amplitude of even & odd filter response.

            sumAn_ThisOrient = sumAn_ThisOrient + An             # Sum of amplitude responses.
            sumE_ThisOrient = sumE_ThisOrient + MatrixEO[:,:,1] # Sum of even filter convolution results.
            sumO_ThisOrient = sumO_ThisOrient + MatrixEO[:,:,0] # Sum of odd filter convolution results.

            # At the smallest scale estimate noise characteristics from the
            # distribution of the filter amplitude responses stored in sumAn.
            # tau is the Rayleigh parameter that is used to describe the
            # distribution.
            if s == 0:
            #     if noiseMethod == -1     # Use median to estimate noise statistics
                tau = np.median(sumAn_ThisOrient) / math.sqrt(math.log(4))#sqrt(E(An))
            #     elseif noiseMethod == -2 # Use mode to estimate noise statistics
            #         tau = rayleighmode(sumAn_ThisOrient(:));
            #     end
                maxAn = An
            else:
                # Record maximum amplitude of components across scales.  This is needed
                # to determine the frequency spread weighting.
                maxAn = np.maximum(maxAn,An)
            # end
        # complete scale loop
        # next section within mother (orientation) loop
        #
        # Accumulate total 3D energy vector data, this will be used to
        # determine overall feature orientation and feature phase/type
        EnergyV[:,:,0] = EnergyV[:,:,0] + sumE_ThisOrient
        EnergyV[:,:,1] = EnergyV[:,:,1] + math.cos(angl)*sumO_ThisOrient
        EnergyV[:,:,2] = EnergyV[:,:,2] + math.sin(angl)*sumO_ThisOrient

        # Get weighted mean filter response vector, this gives the weighted mean
        # phase angle.  paper (11)
        XEnergy = np.sqrt(sumE_ThisOrient**2 + sumO_ThisOrient**2) + epsilon
        MeanE = sumE_ThisOrient / XEnergy
        MeanO = sumO_ThisOrient / XEnergy

        # Now calculate An(cos(phase_deviation) - | sin(phase_deviation)) | by
        # using dot and cross products between the weighted mean filter response
        # vector and the individual filter response vectors at each scale.  This
        # quantity is phase congruency multiplied by An, which we call energy.

        for s in np.arange(NumberScales):
            # Extract even and odd convolution results.
            E = EO[:,:,s,o].real
            O = EO[:,:,s,o].imag
            Energy = Energy + E*MeanE + O*MeanO - np.abs(E*MeanO - O*MeanE)
        ##分配相位一致性方向,只选了最大的尺度？？？
        a = a + EO[:,:,0,o].imag*math.cos(angl)
        b = b + EO[:,:,0,o].imag*math.sin(angl)
        ## Automatically determine noise threshold
        #
        # Assuming the noise is Gaussian the response of the filters to noise will
        # form Rayleigh distribution.  We use the filter responses at the smallest
        # scale as a guide to the underlying noise level because the smallest scale
        # filters spend most of their time responding to noise, and only
        # occasionally responding to features. Either the median, or the mode, of
        # the distribution of filter responses can be used as a robust statistic to
        # estimate the distribution mean and standard deviation as these are related
        # to the median or mode by fixed constants.  The response of the larger
        # scale filters to noise can then be estimated from the smallest scale
        # filter response according to their relative bandwidths.
        #
        # This code assumes that the expected reponse to noise on the phase congruency
        # calculation is simply the sum of the expected noise responses of each of
        # the filters.  This is a simplistic overestimate, however these two
        # quantities should be related by some constant that will depend on the
        # filter bank being used.  Appropriate tuning of the parameter 'k' will
        # allow you to produce the desired output.

        # if noiseMethod >= 0:     % We are using a fixed noise threshold
        #     T = noiseMethod;    % use supplied noiseMethod value as the threshold
        # else:
        # Estimate the effect of noise on the sum of the filter responses as
        # the sum of estimated individual responses (this is a simplistic
        # overestimate). As the estimated noise response at succesive scales
        # is scaled inversely proportional to bandwidth we have a simple
        # geometric sum.
        totalTau = tau * (1 - (1/mult)**NumberScales)/(1-(1/mult))###sigma_G

        # Calculate mean and std dev from tau using fixed relationship
        # between these parameters and tau. See
        # http://mathworld.wolfram.com/RayleighDistribution.html
        EstNoiseEnergyMean = totalTau*math.sqrt(math.pi/2)        # Expected mean:μ_R and std,
        EstNoiseEnergySigma = totalTau*math.sqrt((4-math.pi)/2)   # values of noise energy,sigma_R

        T =  EstNoiseEnergyMean + k*EstNoiseEnergySigma # Noise threshold
        # end

        # Apply noise threshold,  this is effectively wavelet denoising via
        # soft thresholding.
        Energy = np.maximum(Energy - T, 0)

        # Form weighting that penalizes frequency distributions that are
        # particularly narrow.  Calculate fractional 'width' of the frequencies
        # present by taking the sum of the filter response amplitudes and dividing
        # by the maximum amplitude at each point on the image.   If
        # there is only one non-zero component width takes on a value of 0, if
        # all components are equal width is 1.
        width = (sumAn_ThisOrient/(maxAn + epsilon) - 1) / (NumberScales-1)

        # Now calculate the sigmoidal weighting function for this orientation. g:gamma = 10,cutoff = 0.4
        weight = 1.0 / (1 + np.exp( (cutOff - width)*g))

        # Apply weighting to energy and then calculate phase congruency
        PC[:,:,o] = weight*Energy/sumAn_ThisOrient   # Phase congruency for this orientatio

        pcSum = pcSum + PC[:,:,o]

        # Build up covariance data for every point
        covx = PC[:,:,o]*math.cos(angl)
        covy = PC[:,:,o]*math.sin(angl)
        covx2 = covx2 + covx**2
        covy2 = covy2 + covy**2
        covxy = covxy + covx*covy
        # above everyting within orientaiton loop
    pc_orientation = cv2.phase(a,b,angleInDegrees = True)
    return pcSum,pc_orientation

def HOPC_des(gray,coords,NumberScales,NumberAngles,cell_size = 8,bin_size = 8):
    cell_gradient_vector = []
    for row,col in coords:
        cell1 = gray[int(row-cell_size):int(row),int(col-cell_size):int(col)]  
        cell2 = gray[int(row-cell_size):int(row),int(col):int(col+cell_size)]  
        cell3 = gray[int(row):int(row+cell_size),int(col-cell_size):int(col)]  
        cell4 = gray[int(row):int(row+cell_size),int(col):int(col+cell_size)]
        cell_magnitude1,cell_angle1 = PhaseCongruency(cell1,NumberScales,NumberAngles) 
        cell_magnitude2,cell_angle2 = PhaseCongruency(cell2,NumberScales,NumberAngles) 
        cell_magnitude3,cell_angle3 = PhaseCongruency(cell3,NumberScales,NumberAngles) 
        cell_magnitude4,cell_angle4 = PhaseCongruency(cell4,NumberScales,NumberAngles) 
        cell_gradient_vector.append([cell_gradient(cell_magnitude1, cell_angle1,bin_size),cell_gradient(cell_magnitude2, cell_angle2,bin_size),
                                     cell_gradient(cell_magnitude3, cell_angle3,bin_size),cell_gradient(cell_magnitude4, cell_angle4,bin_size)])
    cell_gradient_vector = np.array(cell_gradient_vector)
    cell_gradient_vector = np.reshape(cell_gradient_vector,(coords.shape[0],4*bin_size))
    HOPC_vector = []
    for i in range(cell_gradient_vector.shape[0] ):
        block_vector = []
        block_vector.extend(cell_gradient_vector[i])
        mag = lambda vector: math.sqrt(sum(i ** 2 for i in vector))
        magnitude = mag(block_vector)
        if magnitude != 0:
            normalize = lambda block_vector, magnitude: [element / magnitude for element in block_vector]
            HOPC_vector = normalize(block_vector, magnitude)
        HOPC_vector.append(block_vector)
    return HOPC_vector

def cell_gradient(cell_magnitude, cell_angle,bin_size = 8):
        angle_unit = 360 // bin_size
        orientation_centers = [0] * bin_size
        for i in range(cell_magnitude.shape[0]):
            for j in range(cell_magnitude.shape[1]):
                gradient_strength = cell_magnitude[i][j]
                gradient_angle = cell_angle[i][j]
                min_angle, max_angle, mod = get_closest_bins(gradient_angle,bin_size)
                orientation_centers[min_angle] += (gradient_strength * (1 - (mod / angle_unit)))##给每个方向加权
                orientation_centers[max_angle] += (gradient_strength * (mod / angle_unit))##方向加权
        return orientation_centers
    
def get_closest_bins(pc_angle,bin_size = 8):
        angle_unit = 360 // bin_size
        idx = int(pc_angle / angle_unit)
        mod = pc_angle % angle_unit#加权的偏移量
        if idx == bin_size:
            return idx - 1, (idx) % bin_size, mod
        return idx, (idx + 1) % bin_size, mod 
####



###HOPC template matching   
class HOPC_descriptor():
    def __init__(self, img, cell_size=16, bin_size=8):
        self.img = img
        self.img = np.sqrt(img / float(np.max(img)))
        self.img = self.img * 255
        self.cell_size = cell_size
        self.bin_size = bin_size
        self.angle_unit = 360 // self.bin_size
        self.NumberScales = 4
        self.NumberAngles = 6
        assert type(self.bin_size) == int, "bin_size should be integer,"
        assert type(self.cell_size) == int, "cell_size should be integer,"
        assert type(self.angle_unit) == int, "bin_size should be divisible by 360"

    def extract(self):
        height, width = self.img.shape
        pc_magnitude, pc_angle = self.PhaseCongruency()
        pc_magnitude = abs(pc_magnitude)
        cell_pc_vector = np.zeros((height // self.cell_size, width // self.cell_size, self.bin_size))
        for i in range(cell_pc_vector.shape[0]):
            for j in range(cell_pc_vector.shape[1]):
                cell_magnitude = pc_magnitude[i * self.cell_size:(i + 1) * self.cell_size,
                                 j * self.cell_size:(j + 1) * self.cell_size]
                cell_angle = pc_angle[i * self.cell_size:(i + 1) * self.cell_size,
                             j * self.cell_size:(j + 1) * self.cell_size]
                cell_pc_vector[i][j] = self.cell_pc(cell_magnitude, cell_angle)
 
        hopc_image = self.render_pc(np.zeros([height, width]), cell_pc_vector)
        hopc_vector = []
        for i in range(cell_pc_vector.shape[0] - 1):
            for j in range(cell_pc_vector.shape[1] - 1):
                block_vector = []
                block_vector.extend(cell_pc_vector[i][j])
                block_vector.extend(cell_pc_vector[i][j + 1])
                block_vector.extend(cell_pc_vector[i + 1][j])
                block_vector.extend(cell_pc_vector[i + 1][j + 1])
                mag = lambda vector: math.sqrt(sum(i ** 2 for i in vector))
                magnitude = mag(block_vector)
                if magnitude != 0:
                    normalize = lambda block_vector, magnitude: [element / magnitude for element in block_vector]
                    block_vector = normalize(block_vector, magnitude)
                hopc_vector.append(block_vector)
        return hopc_vector, hopc_image
    def PhaseCongruency(self):

        # nscale           4    - Number of wavelet scales, try values 3-6
        # norient          6    - Number of filter orientations.
        # minWaveLength    3    - Wavelength of smallest scale filter.
        # mult             2.1  - Scaling factor between successive filters.
        # sigmaOnf         0.55 - Ratio of the standard deviation of the Gaussian
        #                         describing the log Gabor filter's transfer function
        #                         in the frequency domain to the filter center frequency.
        # k                2.0  - No of standard deviations of the noise energy beyond
        #                         the mean at which we set the noise threshold point.
        #                         You may want to vary this up to a value of 10 or
        #                         20 for noisy images
        # cutOff           0.5  - The fractional measure of frequency spread
        #                         below which phase congruency values get penalized.
        # g                10   - Controls the sharpness of the transition in
        #                         the sigmoid function used to weight phase
        #                         congruency for frequency spread.
        # noiseMethod      -1   - Parameter specifies method used to determine
        #                         noise statistics.
        #                           -1 use median of smallest scale filter responses
        #                           -2 use mode of smallest scale filter responses
        #                            0+ use noiseMethod value as the fixed noise threshold
        minWaveLength = 3
        mult = 2.1
        sigmaOnf = 0.55
        k = 2.0
        cutOff = 0.5
        g = 10
        noiseMethod = -1
        InputImage = self.img
        NumberScales = self.NumberScales
        NumberAngles = self.NumberAngles
    
        epsilon = .0001 # Used to prevent division by zero.
    
    
        f_cv = cv2.dft(np.float32(InputImage),flags=cv2.DFT_COMPLEX_OUTPUT)
    
        #------------------------------
        nrows, ncols = InputImage.shape
        zero = np.zeros((nrows,ncols))
        EO = np.zeros((nrows,ncols,NumberScales,NumberAngles),dtype=complex)
        PC = np.zeros((nrows,ncols,NumberAngles))
        covx2 = np.zeros((nrows,ncols))
        covy2 = np.zeros((nrows,ncols))
        covxy = np.zeros((nrows,ncols))
        EnergyV = np.zeros((nrows,ncols,3))
        pcSum = np.zeros((nrows,ncols))
    
        #分配方向
        a = np.zeros((nrows,ncols))
        b = np.zeros((nrows,ncols))
    
        # Matrix of radii
        cy = math.floor(nrows/2)
        cx = math.floor(ncols/2)
        y, x = np.mgrid[0:nrows, 0:ncols]
        y = (y-cy)/nrows
        x = (x-cx)/ncols
    
        radius = np.sqrt(x**2 + y**2)
        radius[cy, cx] = 1
    
        # Matrix values contain polar angle.
        # (note -ve y is used to give +ve anti-clockwise angles)
        theta = np.arctan2(-y, x)
        sintheta = np.sin(theta)
        costheta = np.cos(theta)
    
        # Initialise set of annular bandpass filters
        #  Here I use the method of scale selection from the code I used to generate
        #   stimuli for my latest experiments (spatial feature scaling):
        #   /Users/carl/Studies/Face_Projects/features_wavelet
        #NumberScales = 3 # should be odd
        annularBandpassFilters = np.empty((nrows,ncols,NumberScales))
    
    
        # Number of filter orientations.
        #NumberAngles = 6
        """ Ratio of angular interval between filter orientations and the standard deviation
            of the angular Gaussian function used to construct filters in the freq. plane.
        """
    
        # The following implements the log-gabor transfer function
        """ From http://www.peterkovesi.com/matlabfns/PhaseCongruency/Docs/convexpl.html
            The filter bandwidth is set by specifying the ratio of the standard deviation
            of the Gaussian describing the log Gabor filter's transfer function in the
            log-frequency domain to the filter center frequency. This is set by the parameter
            sigmaOnf . The smaller sigmaOnf is the larger the bandwidth of the filter.
            I have not worked out an expression relating sigmaOnf to bandwidth, but
            empirically a sigmaOnf value of 0.75 will result in a filter with a bandwidth
            of approximately 1 octave and a value of 0.55 will result in a bandwidth of
            roughly 2 octaves.
        """
        # sigmaOnf = 0.74  # approximately 1 octave
        # sigmaOnf = 0.55  # approximately 2 octaves
        """ From Wilson, Loffler and Wilkinson (2002 Vision Research):
            The bandpass filtering alluded to above was used because of ubiquitous evidence
            that face discrimination is optimal within a 2.0 octave (at half amplitude)
            bandwidth centered upon 8–13 cycles per face width (Costen et al., 1996;
            Fiorentini et al., 1983; Gold et al., 1999; Hayes et al., 1986; Näsänen, 1999).
            We therefore chose a radially symmetric filter with a peak frequency of 10.0
            cycles per mean face width and a 2.0 octave bandwidth described by a difference
             of Gaussians (DOG):"""
    
        # Lowpass filter to remove high frequency 'garbage'
        filterorder = 15  # filter 'sharpness'
        cutoff = .45
        normradius = radius / (abs(x).max()*2)
        lowpassbutterworth = 1.0 / (1.0 + (normradius / cutoff)**(2*filterorder))
        #
        # Note: lowpassbutterworth is currently DC centered.
    
        for s in np.arange(NumberScales):
            wavelength = minWaveLength*mult**s
            fo = 1.0/wavelength                  # Centre frequency of filter.
            logGabor = np.exp((-(np.log(radius/fo))**2) / (2 * math.log(sigmaOnf)**2))
            annularBandpassFilters[:,:,s] = logGabor*lowpassbutterworth  # Apply low-pass filter
            annularBandpassFilters[cy,cx,s] = 0          # Set the value at the 0 frequency point of the filter
                                                         # back to zero (undo the radius fudge).
    
        # main loop
        for o in np.arange(NumberAngles):
            # Construct the angular filter spread function
            angl = o*math.pi/NumberAngles # Filter angle.
            # For each point in the filter matrix calculate the angular distance from
            # the specified filter orientation.  To overcome the angular wrap-around
            # problem sine difference and cosine difference values are first computed
            # and then the atan2 function is used to determine angular distance.
    
            # % Scale theta so that cosine spread function has the right wavelength and clamp to pi
            # dtheta = min(dtheta*norient/2,pi);
            # % The spread function is cos(dtheta) between -pi and pi.  We add 1,
            # % and then divide by 2 so that the value ranges 0-1
            """ For each point in the filter matrix calculate the angular distance from the
                specified filter orientation.  To overcome the angular wrap-around problem
                sine difference and cosine difference values are first computed and then
                the atan2 function is used to determine angular distance.
            """
            ds = sintheta * math.cos(angl) - costheta * math.sin(angl)      # Difference in sine.
            dc = costheta * math.cos(angl) + sintheta * math.sin(angl)      # Difference in cosine.
            dtheta = np.abs(np.arctan2(ds,dc))                              # Absolute angular distance.
    
            # Scale theta so that cosine spread function has the right wavelength
            #   and clamp to pi
            dtheta = np.minimum(dtheta*NumberAngles/2, math.pi)
    
            #spread = np.exp((-dtheta**2) / (2 * thetaSigma**2));  # Calculate the angular
                                                                  # filter component.
            # The spread function is cos(dtheta) between -pi and pi.  We add 1,
            #   and then divide by 2 so that the value ranges 0-1
            spread = (np.cos(dtheta)+1)/2
    
            sumE_ThisOrient   = np.zeros((nrows,ncols))  # Initialize accumulator matrices.
            sumO_ThisOrient   = np.zeros((nrows,ncols))
            sumAn_ThisOrient  = np.zeros((nrows,ncols))
            Energy            = np.zeros((nrows,ncols))
    
            maxAn = []
            for s in np.arange(NumberScales):
                filter = annularBandpassFilters[:,:,s] * spread # Multiply radial and angular
                                                                # components to get the filter.
    
                criticalfiltershift = np.fft.ifftshift( filter )
                criticalfiltershift_cv = np.empty((nrows, ncols, 2))
                for ip in range(2):
                    criticalfiltershift_cv[:,:,ip] = criticalfiltershift
    
                # Convolve image with even and odd filters returning the result in EO
                MatrixEO = cv2.idft( criticalfiltershift_cv * f_cv )
                EO[:,:,s,o] = MatrixEO[:,:,1] + 1j*MatrixEO[:,:,0]
    
                An = cv2.magnitude(MatrixEO[:,:,0], MatrixEO[:,:,1])    # Amplitude of even & odd filter response.
    
                sumAn_ThisOrient = sumAn_ThisOrient + An             # Sum of amplitude responses.
                sumE_ThisOrient = sumE_ThisOrient + MatrixEO[:,:,1] # Sum of even filter convolution results.
                sumO_ThisOrient = sumO_ThisOrient + MatrixEO[:,:,0] # Sum of odd filter convolution results.
    
                # At the smallest scale estimate noise characteristics from the
                # distribution of the filter amplitude responses stored in sumAn.
                # tau is the Rayleigh parameter that is used to describe the
                # distribution.
                if s == 0:
                #     if noiseMethod == -1     # Use median to estimate noise statistics
                    tau = np.median(sumAn_ThisOrient) / math.sqrt(math.log(4))#sqrt(E(An))
                #     elseif noiseMethod == -2 # Use mode to estimate noise statistics
                #         tau = rayleighmode(sumAn_ThisOrient(:));
                #     end
                    maxAn = An
                else:
                    # Record maximum amplitude of components across scales.  This is needed
                    # to determine the frequency spread weighting.
                    maxAn = np.maximum(maxAn,An)
                # end
            # complete scale loop
            # next section within mother (orientation) loop
            #
            # Accumulate total 3D energy vector data, this will be used to
            # determine overall feature orientation and feature phase/type
            EnergyV[:,:,0] = EnergyV[:,:,0] + sumE_ThisOrient
            EnergyV[:,:,1] = EnergyV[:,:,1] + math.cos(angl)*sumO_ThisOrient
            EnergyV[:,:,2] = EnergyV[:,:,2] + math.sin(angl)*sumO_ThisOrient
    
            # Get weighted mean filter response vector, this gives the weighted mean
            # phase angle.  paper (11)
            XEnergy = np.sqrt(sumE_ThisOrient**2 + sumO_ThisOrient**2) + epsilon
            MeanE = sumE_ThisOrient / XEnergy
            MeanO = sumO_ThisOrient / XEnergy
    
            # Now calculate An(cos(phase_deviation) - | sin(phase_deviation)) | by
            # using dot and cross products between the weighted mean filter response
            # vector and the individual filter response vectors at each scale.  This
            # quantity is phase congruency multiplied by An, which we call energy.
    
            for s in np.arange(NumberScales):
                # Extract even and odd convolution results.
                E = EO[:,:,s,o].real
                O = EO[:,:,s,o].imag
                Energy = Energy + E*MeanE + O*MeanO - np.abs(E*MeanO - O*MeanE)
            ##分配相位一致性方向,只选了最大的尺度？？？
            a = a + EO[:,:,0,o].imag*math.cos(angl)
            b = b + EO[:,:,0,o].imag*math.sin(angl)
            ## Automatically determine noise threshold
            #
            # Assuming the noise is Gaussian the response of the filters to noise will
            # form Rayleigh distribution.  We use the filter responses at the smallest
            # scale as a guide to the underlying noise level because the smallest scale
            # filters spend most of their time responding to noise, and only
            # occasionally responding to features. Either the median, or the mode, of
            # the distribution of filter responses can be used as a robust statistic to
            # estimate the distribution mean and standard deviation as these are related
            # to the median or mode by fixed constants.  The response of the larger
            # scale filters to noise can then be estimated from the smallest scale
            # filter response according to their relative bandwidths.
            #
            # This code assumes that the expected reponse to noise on the phase congruency
            # calculation is simply the sum of the expected noise responses of each of
            # the filters.  This is a simplistic overestimate, however these two
            # quantities should be related by some constant that will depend on the
            # filter bank being used.  Appropriate tuning of the parameter 'k' will
            # allow you to produce the desired output.
    
            # if noiseMethod >= 0:     % We are using a fixed noise threshold
            #     T = noiseMethod;    % use supplied noiseMethod value as the threshold
            # else:
            # Estimate the effect of noise on the sum of the filter responses as
            # the sum of estimated individual responses (this is a simplistic
            # overestimate). As the estimated noise response at succesive scales
            # is scaled inversely proportional to bandwidth we have a simple
            # geometric sum.
            totalTau = tau * (1 - (1/mult)**NumberScales)/(1-(1/mult))###sigma_G
    
            # Calculate mean and std dev from tau using fixed relationship
            # between these parameters and tau. See
            # http://mathworld.wolfram.com/RayleighDistribution.html
            EstNoiseEnergyMean = totalTau*math.sqrt(math.pi/2)        # Expected mean:μ_R and std,
            EstNoiseEnergySigma = totalTau*math.sqrt((4-math.pi)/2)   # values of noise energy,sigma_R
    
            T =  EstNoiseEnergyMean + k*EstNoiseEnergySigma # Noise threshold
            # end
    
            # Apply noise threshold,  this is effectively wavelet denoising via
            # soft thresholding.
            Energy = np.maximum(Energy - T, 0)
    
            # Form weighting that penalizes frequency distributions that are
            # particularly narrow.  Calculate fractional 'width' of the frequencies
            # present by taking the sum of the filter response amplitudes and dividing
            # by the maximum amplitude at each point on the image.   If
            # there is only one non-zero component width takes on a value of 0, if
            # all components are equal width is 1.
            width = (sumAn_ThisOrient/(maxAn + epsilon) - 1) / (NumberScales-1)
    
            # Now calculate the sigmoidal weighting function for this orientation. g:gamma = 10,cutoff = 0.4
            weight = 1.0 / (1 + np.exp( (cutOff - width)*g))
    
            # Apply weighting to energy and then calculate phase congruency
            PC[:,:,o] = weight*Energy/sumAn_ThisOrient   # Phase congruency for this orientatio
    
            pcSum = pcSum + PC[:,:,o]
    
            # Build up covariance data for every point
            covx = PC[:,:,o]*math.cos(angl)
            covy = PC[:,:,o]*math.sin(angl)
            covx2 = covx2 + covx**2
            covy2 = covy2 + covy**2
            covxy = covxy + covx*covy
            # above everyting within orientaiton loop
        pc_orientation = cv2.phase(a,b,angleInDegrees = True)
        return pcSum,pc_orientation
    

    def cell_pc(self, cell_magnitude, cell_angle):
        orientation_centers = [0] * self.bin_size
        for i in range(cell_magnitude.shape[0]):
            for j in range(cell_magnitude.shape[1]):
                gradient_strength = cell_magnitude[i][j]
                gradient_angle = cell_angle[i][j]
                min_angle, max_angle, mod = self.get_closest_bins(gradient_angle)
                orientation_centers[min_angle] += (gradient_strength * (1 - (mod / self.angle_unit)))
                orientation_centers[max_angle] += (gradient_strength * (mod / self.angle_unit))
        return orientation_centers

    def get_closest_bins(self, gradient_angle):
        idx = int(gradient_angle / self.angle_unit)
        mod = gradient_angle % self.angle_unit
        if idx == self.bin_size:
            return idx - 1, (idx) % self.bin_size, mod
        return idx, (idx + 1) % self.bin_size, mod

    def render_pc(self, image, cell_gradient):
        '''可视化cell'''
        cell_width = self.cell_size / 2
        max_mag = np.array(cell_gradient).max()
        for x in range(cell_gradient.shape[0]):
            for y in range(cell_gradient.shape[1]):
                cell_grad = cell_gradient[x][y]
                cell_grad /= max_mag
                angle = 0
                angle_gap = self.angle_unit
                for magnitude in cell_grad:
                    angle_radian = math.radians(angle)
                    x1 = int(x * self.cell_size + magnitude * cell_width * math.cos(angle_radian))
                    y1 = int(y * self.cell_size + magnitude * cell_width * math.sin(angle_radian))
                    x2 = int(x * self.cell_size - magnitude * cell_width * math.cos(angle_radian))
                    y2 = int(y * self.cell_size - magnitude * cell_width * math.sin(angle_radian))
                    cv2.line(image, (y1, x1), (y2, x2), int(255 * math.sqrt(magnitude)))
                    angle += angle_gap
        return image   
    
if __name__ == '__main__':
    img = cv2.imread('D:/Study/Master_project/code/image/GPR/GPR2_s1.png',0);
    HOPC = HOPC_descriptor(img, cell_size=16, bin_size=8)
    vector, image_hopc = HOPC.extract()
    plt.figure();plt.imshow(image_hopc, cmap=plt.cm.gray)
    
    img = cv2.imread('D:/Study/Master_project/code/image/GPR/GPR4_s1.png',0);
    HOPC = HOPC_descriptor(img, cell_size=16, bin_size=8)
    vector, image1_hopc = HOPC.extract()
    plt.figure();plt.imshow(image1_hopc, cmap=plt.cm.gray)