function Image_Out = ScottRecon3D(ImageSize,data,traj)
%% A Function written to reconstruct Images when K-space data and trajectories are passed to it
% Uses Scott Robertson's reconstruction code - This just makes it more
% modular and easy to implement - This is for 3D data.
% 
% ImageSize - Output image matrix size -If Scalar, Isotropic, ImageSize
%                                       If 3-Vector,
%
% data - KSpace Data in column vector (N x 1)
%
% traj - point in kspace corresponding to the data vector - columns for
% x,y, and z. (N x 3)
%
% Author information:
% Reconstruction script: Scott H. Robertson
% Optimzation for 2D spiral: Joseph W. Plummer

% Make sure ImageSize is Meaningful
if numel(ImageSize)==1
   ImageSize = [ImageSize(1) ImageSize(1) ImageSize(1)]; 
elseif numel(ImageSize) ~= 3
   error('ImageSize needs to be either a scalar or 3-d vector.'); 
end

kernel.sharpness = 0.45; 
kernel.extent = 6*kernel.sharpness; 
overgrid_factor = 2;
if numel(ImageSize) == 1
    output_image_size = ImageSize*[1 1 1];
else
    output_image_size = ImageSize;
end
nDcfIter = 12;
deapodizeImage = false;
nThreads = 10;
cropOvergriddedImage = true();
verbose = true();

%  Choose kernel, proximity object, and then create system model
kernelObj = Recon.SysModel.Kernel.Gaussian(kernel.sharpness, kernel.extent, verbose);
%kernelObj = Recon.SysModel.Kernel.KaiserBessel(kernel.sharpness, kernel.extent, verbose);
%kernelObj = Recon.SysModel.Kernel.Sinc(kernel.sharpness, kernel.extent, verbose);

proxObj = Recon.SysModel.Proximity.L2Proximity(kernelObj, verbose);
%proxObj = Recon.SysModel.Proximity.L1Proximity(kernelObj, verbose);
clear kernelObj;
systemObj = Recon.SysModel.MatrixSystemModel(traj, overgrid_factor, ...
    output_image_size, proxObj, verbose);

% Choose density compensation function (DCF)
dcfObj = Recon.DCF.Iterative(systemObj, nDcfIter, verbose);
%dcfObj = Recon.DCF.Voronoi(traj, header, verbose);
%dcfObj = Recon.DCF.Analytical3dRadial(traj, verbose);
%dcfObj = Recon.DCF.Unity(traj, verbose);

% Choose Reconstruction Model
reconObj = Recon.ReconModel.LSQGridded(systemObj, dcfObj, verbose);
clear modelObj;
clear dcfObj;
reconObj.crop = cropOvergriddedImage;
reconObj.deapodize = deapodizeImage;

% Reconstruct image using trajectories in pixel units
Image_Out = reconObj.reconstruct(data, traj);
end

