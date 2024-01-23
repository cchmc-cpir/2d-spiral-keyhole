function Image_Out = ScottRecon3D_Healthy_optimized(ImageSize,data,traj,PixelShift)
%% A Function written to reconstruct Images when K-space data and trajectories are passed to it
% Uses Scott Robertson's reconstruction code - This just makes it more
% modular and easy to implement - This is for 3D data
% 
% ImageSize - Output image matrix size -If Scalar, Isotropic, ImageSize
%                                       If 3-Vector,
%
% data - KSpace Data in column vector (N x 1)
%
% traj - point in kspace corresponding to the data vector - columns for
% x,y, and z. (N x 3)
%
% PixelShift = array of size 3 with pixels to shift by [a(-)p(+); r(-)l(+); h(+)f(-)]

% Make sure ImageSize is meaningful:
if numel(ImageSize) == 1
   ImageSize = [ImageSize(1) ImageSize(1) ImageSize(1)]; 
elseif numel(ImageSize) ~= 3
   error('ImageSize needs to be either a scalar or 3-d vector.'); 
end
% Ensure the pixel shift is correct:
if exist('PixelShift','var') == 0 % If not passed, set to 0's
    PixelShift = [0, 0, 0];
end

%% Gridding parameters:
kernel.sharpness = 0.35;
kernel.extent = 6*kernel.sharpness;
overgrid_factor = 2;
output_image_size = ImageSize;
nDcfIter = 10;
deapodizeImage = true(); 
nThreads = 10;
cropOvergriddedImage = true();
verbose = true();

% Account for true res
traj = traj*1;
highs = any(traj>0.5,2); % Recon can't handle greater than 0.5 mag
lows = any(traj<-0.5,2);
removes = any([highs lows],2);% Remove them
traj = traj(~removes,:);
data = data(~removes);

%% Begin reconstruction:
%  Choose kernel, proximity object, and then create system model
kernelObj = Recon.SysModel.Kernel.Gaussian(kernel.sharpness, kernel.extent, verbose);
%kernelObj = Recon.SysModel.Kernel.KaiserBessel(kernel.sharpness, kernel.extent, verbose);
%kernelObj = Recon.SysModel.Kernel.Sinc(kernel.sharpness, kernel.extent, verbose);

proxObj = Recon.SysModel.Proximity.L2Proximity(kernelObj, verbose);
% proxObj = Recon.SysModel.Proximity.L1Proximity(kernelObj, verbose);
clear kernelObj;
systemObj = Recon.SysModel.MatrixSystemModel(traj, overgrid_factor, ...
    output_image_size, proxObj, verbose);

% Choose density compensation function (DCF)
dcfObj = Recon.DCF.Iterative(systemObj, nDcfIter, verbose);
% dcfObj = Recon.DCF.Voronoi(traj, header, verbose);
% dcfObj = Recon.DCF.Analytical3dRadial(traj, verbose);
% dcfObj = Recon.DCF.Unity(traj, verbose);

% Choose Reconstruction Model
reconObj = Recon.ReconModel.LSQGridded(systemObj, dcfObj, verbose);
clear modelObj;
clear dcfObj;
reconObj.PixelShift = PixelShift;
reconObj.crop = cropOvergriddedImage;
reconObj.deapodize = deapodizeImage;

% Reconstruct image using trajectories in pixel units
Image_Out = reconObj.reconstruct(data, traj);
Image_Out = rot90(flip(Image_Out));
