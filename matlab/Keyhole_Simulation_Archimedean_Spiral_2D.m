%% TITLE SECTION:
%%%% Keyhole Reconstruction Simulation for 2D Archimedean Spirals %%%%
%
%%%% Date: %%%%
% 07/19/2021 - Complete provisional copy and send to coauthors
%
%%%% Agenda: %%%%
% Produce accurate flip angle maps using keyhole reconstruction. Then, use
% these flip angle maps to correct the image for areas where the flip angle
% deviated from the intended angle. 
%
%%%% Author information: %%%%
% Joseph W. Plummer - Grad student in CPIR - joseph.plummer@cchmc.org
% Scott H. Robertson - Ex. student at Duke - https://doi.org/10.1002/cmr.a.21352
%
%%%% Additional credits: %%%%
% Mariah Costa - uncertainty calculations
% Abdullah Bdaiwi - advice and theory
% Matthew Willmering - advice and theory
% Peter Niedbalski/Jim Pipe - spiral coordinate conversion script
%
%%%% General information: %%%%
% Can run as one whole script, or section by section (recommended for
% learning purposes).
%
%
clc; clear all; close all;

%% 1) First, set some parameters:
% Scanner side: (units  ms, kHz, m, mT):
N_spirals = 34; % Number of spiral interleaves
fovxy = 0.350; % Give yourself extra room in the FOV to account for reduced image fraction
resxy = 0.003; 
fovz = 0.015;
mslew = 150; % 400
mgrad = 32.5; % 22.5
dwell = 0.01; 

% Reconstruction side:
N_keys = 2; % Number of keys.
r_key = 0.12; % Key radius - range from 0 to 1. Example: 0.5 = 50% max radius.

% For fully sampled keyhole reconstruction, define R according to:
R = @(r_key, N_keys) 1./(r_key.*(N_keys - 1) + 1);
R = R(r_key, N_keys);

% Or manually define R...
R = 0.5; % Undersampling factor - range from 0.25 to 2.

% Advanced scanner settings: 
xdely = 0; ydely = 0; zdely = 0;
gamma = 11.776227;  resz = 0; stype = 0; taper = 0; hubs = 0; alpha0 = 0; 
rebin = 0; us_0 = 0; us_1 = 1; us_r0 = R; us_r = 1; utype = 2; mgfrq = 0.94; 
t2mch = 0; slper = 0; gtype = 2; spinout = 0; numCalPnts = 0; slwin = 100;
% #define spUSTYPE_LINEAR 0
% #define spUSTYPE_QUAD   1
% #define spUSTYPE_HANN   2
% #define spUSTYPE_NONE   3

% Visualize the variable density undersample function:
% undersamplefigure = Functions.undersamplefunction(us_r0, us_r, us_0, us_1, utype)

%% 2) Convert these parameters into coordinates using c/c++ functions:
% Source: BNI coordinate conversion generated by Peter Niedbalski
coords = +Functions.bnispiralgen_vm_matched(dwell, xdely, ydely, zdely, mslew, mgrad,...
                               gamma, fovxy, fovz, resxy, resz, stype,...
                               N_spirals, taper, hubs, alpha0, rebin, us_0,...
                               us_1, us_r0, us_r, utype, mgfrq, t2mch, slper,...
                               gtype, spinout, numCalPnts, slwin);
coords = squeeze(coords(:,:,:,1));

% Extract the dimension sizes:
N_samp = size(coords,2);
N_spirals = size(coords,3);

%% 3) Determine an output image size:
ImageSize = round(fovxy/resxy);  
ImageSize = 2*floor(ImageSize/2); % Ensure even number
OverSize = 1; % Overgridding factor for FFT/trajectories (1 to 10)
GenSize = ImageSize*OverSize; 
ImSize = [ImageSize,ImageSize,1]; % Single slice, but in 3 dimensions.

%% 4) Generate a phantom object:
% Phantom shape 1:
% MySlice = zeros(GenSize, GenSize);
% MySlice(10:GenSize-10,10:GenSize-10) = 1;

% Phantom shape 2
MySlice = Functions.lungphantom2D(GenSize);

% Phantom shape 3
% MySlice = phantom('Modified Shepp-Logan',0.8*GenSize);
% MySlice = padarray(MySlice,[0.1*GenSize 0.1*GenSize],0,'both');
% MySlice = MySlice.*10000;
% MySlice(MySlice < 0) = 100;
% MySlice = rot90(MySlice,2);
% MySlice = flip(MySlice,2);

% Phantom shape 4
% IMFrac = 0.5; % Fraction of image occupied by object
% % NOTE: Recommend using a fraction <= 0.5 as larger objects cause FOV
% % artifacts (ghosting). Alternatively, pad object with zeroes.
% E = [1 IMFrac IMFrac 0 0 0];
% MySlice = phantom(E, GenSize);

% Normalize phantom:
MySlice = MySlice./max(MySlice(:)); 

% Add a layer of random defect pixels: (arbitrary method)
VDP = 0; % For example, 5 = 5% VDP.
defects = 100/VDP*rand(ImageSize, ImageSize);
defects(defects > 1) = 1;
defects(:, ImageSize/2:end) = 1;
defects = imopen(defects,strel('ball',7,7));
MySlice = MySlice.*defects;

% View the image:
f1 = figure('Name','Phantom Image')
imshow(MySlice)
title('Phantom Image', 'color', 'k')
pause(1)

%% 5) Create a binary mask for the image:
% Mask = phantom(E, ImageSize); 
Mask = ones(ImageSize);
Mask = MySlice > 0;
Mask = Mask./max(Mask(:)); % Normalize it to 1s and 0s
Mask(Mask == 0) = 0.05;

%% 6) Define the applied Flip angle:
% (Remember, the applied flip angle ~= true flip angle because of field
% inhomogeneity.) 

% Arbitrarily define the general flip angle. We will use this expression:
FAOpt = round(atand(sqrt(2/N_spirals)));

% Generate a flip angle map:
% [FAMap, FA, pattern] = Functions.flipanglemap(FAOpt, GenSize, "Quadrant");
% [FAMap, FA, pattern] = Functions.flipanglemap(FAOpt, GenSize, "Corner");
% [FAMap, FA, pattern] = Functions.flipanglemap(FAOpt, GenSize, "Leftright");
% [FAMap, FA, pattern] = Functions.flipanglemap(FAOpt, GenSize, "Trigonometry");
[FAMap, FA, pattern] = Functions.flipanglemap(FAOpt, GenSize, "Peaks");
% [FAMap, FA, pattern] = Functions.flipanglemap(FAOpt, GenSize, "Cross");
% [FAMap, FA, pattern] = Functions.flipanglemap(FAOpt, GenSize, "Windmill");

% Visualize the applied flip angle map:
f2 = figure('Name','Applied Flip Angle Map')
CMap = parula;
CMap(1,:) = [0 0 0];
imagesc(FAMap)
axis square
axis off
colormap(gca, CMap)
caxis(gca,[0 max(FA)+10]); colorbar;
title('Applied Flip Angle Map')
pause(1)

%% 7) Define and manipulate the trajectory coordinates:
traj = coords(:,:,:);
% First dimension gives x,y,z coords of traj. Second dimension is the
% sample number of each trajectory. Third dimension is the trajectory
% number (RF pulse number). 

% The bnispiralgen_vm_matched.cpp function does not produce spirals that
% are separated by the golden angle formula as in Jim Pipe's 2011 paper.
% Instead, each spiral is separated by (2*pi / narms) and rotates
% clockwise. Let us try and perform the rotations ourselves using the
% rotation matrix. We can consider lots of different rotational methods, to
% imitate the output of the MR scanner.

% Default settings:
beta = zeros(1,size(traj,3)); % Default
beta_correction = "Linear (default)"; % Default
for ii = 1:size(traj,3)
    % Generate a list of other possible angle corrections. 
    % SELECT ONE:
%     
    % Golden angle:
%     beta(ii) = ((ii - 1)*pi*(3 - sqrt(5))) + ((ii - 1)*2*pi/N_spirals);
%     beta_correction = "Golden";

    % Linear angle, two full revolutions:
%    beta(ii) = ((ii - 1)*2*pi/(narms/2)) + ((ii - 1)*2*pi/narms);
%    beta_correction = "Double linear";
    
    % No angle between spirals (i.e. continuously repeat):
%     beta(ii) = ((ii - 1)*2*pi/N_spirals);
%     beta_correction = "None";

    % Reverse spiral direction:
%     beta(ii) = 2*((ii - 1)*2*pi/narms); 
%   beta_correction = "Negative Linear";
    
    % Rotate the trajectory x and y coords about the golden angle
    % correction (beta). (Rotation matrix around z).
    Rot(:,:,ii) = [cos(beta(ii)) -sin(beta(ii)) 0;...
        sin(beta(ii)) cos(beta(ii)) 0; 0 0 1];  
    for jj = 1:size(traj,2)
        traj(:,jj,ii) = Rot(:,:,ii)*traj(:,jj,ii);
    end
end

% If you want to oversample by adding more spirals:
% oversample = 1; % default
% oversample = 0.8;
% N_spirals_os = ceil(N_spirals./oversample);
% for ii = 1:N_spirals_os
%     % Oversample by adding more spirals:
%     beta(ii) = ((ii - 1)*2*pi/N_spirals_os);
%     beta_correction = "Added spirals";
%     
%     % Rotate the trajectory x and y coords about the angle
%     % correction (beta). (Rotation matrix around z).
%     Rot(:,:,ii) = [cos(beta(ii)) -sin(beta(ii)) 0;...
%         sin(beta(ii)) cos(beta(ii)) 0; 0 0 1];  
%     for jj = 1:size(traj,2)
%         traj(:,jj,ii) = Rot(:,:,ii)*traj(:,jj,ii);
%     end
% end

% We may also want to rearrange the order of the spirals:
% SELECT ONE:
[order, ordering_type] = Functions.philipsreorder(N_spirals, "Ascending"); % Default
% [order, ordering_type] = Functions.philipsreorder(N_shots, "Descending");
% [order, ordering_type] = Functions.philipsreorder(N_shots, "Two-way");
% [order, ordering_type] = Functions.philipsreorder(N_spirals, "Skipped");
[order, ordering_type] = Functions.philipsreorder(N_spirals, "Mixed");

% Reorder the trajectories:
traj_dummy(:,:,:) = traj(:,:,:);
for ii = 1:size(traj,3)
     traj(:,:,ii) = traj_dummy(:,:,order(ii));
end
sum(sum(sum(traj-traj_dummy)));

% Clear old variables if used:
clear traj_dummy

% Each of the trajectories are small. Scale them up to very large, so that
% we can be more 'accurate' when mapping the spiral trajectory locations
% onto a cartesian sampling grid (using rounding or interpolating) for this
% simulation only. We don't do this for real images!
SampTraj = (traj*ImageSize + GenSize/2 + 1); % Ensure it starts at center of k-space

% Save the trajectory animations as a video:
% myVideo = VideoWriter('Sampling_coords'); % Open video file
% myVideo.FrameRate = 2;  % Can adjust this, 5 - 10 works well 
% open(myVideo)

% Visualize the coordinate rounding process:
f3 = figure('Name','Rounded trajectories')
set(gcf,'color','w');
for i = 1:size(traj,3) % Third dimension represents each RF pulse
    pause(0.1)
    plot(squeeze(SampTraj(1,:,i)),squeeze(SampTraj(2,:,i)),'b.')
    hold on
    axis square
    axis off
    title('Rounded trajectory coordinates')
    xlim([0 ImageSize])
    ylim([0 ImageSize])
            
    % Update for video:
%     frame = getframe(gcf); %get frame
%     writeVideo(myVideo, frame);
end
hold off
% close(myVideo)

% Close figures:
close(f1)
close(f2)
close(f3)

%% 8) Simulate HP Xe decay: 
% Decay according to magnetization decay equations:
DecayMap = cosd(FAMap);
SampPhant = sind(FAMap).*MySlice;

% Initialize k-space data:
KSpaceDecay = zeros(N_samp, N_spirals);

% Apply decay for each RF pulse.
for RF_indx = 1:N_spirals 
    Decay(:,:,RF_indx) = DecayMap.^(RF_indx - 1); 
    PhantDecay(:,:,RF_indx) = SampPhant .* Decay(:,:,RF_indx); 
    KSpaceSampDecay(:,:,RF_indx) = fftshift(fft2(ifftshift(...
        PhantDecay(:,:,RF_indx))));

    % After each RF pulse, sample the k-space data:
%     for Samp_indx = 1:size(SampTraj,2)
        % Because we can't find the exact point in spiral coordinates,
        % sample each spiral point at a cartesian location using rounded
        % coordinates. This is why we used overgridding above.
        
        % Sample each k-space signal value from its rounded spiral
        % location: (remember, we only need each x and y coordinate of each
        % RF pulse being applied).
%         KSpaceDecay(Samp_indx, RF_indx) = KSpaceSampDecay(SampTraj(...
%             1,Samp_indx,RF_indx), SampTraj(2,Samp_indx,RF_indx), RF_indx);
        KSpaceDecay(:, RF_indx) = interp2(KSpaceSampDecay(:,:,RF_indx),SampTraj(1,:,RF_indx),SampTraj(2,:,RF_indx));
%     end
    
end

%  Add random Gaussian noise
noisesd = 1E-1; 
noise = complex(normrnd(0,noisesd,size(KSpaceDecay)), ...
    normrnd(0,noisesd,size(KSpaceDecay)));
KSpaceDecay = KSpaceDecay + noise;

%% 9) Begin the keyhole segregation and cutoffs:
% Convert spiral coords into radial values:
traj_rad = squeeze(sqrt(traj(1,:,:).^2 + traj(2,:,:).^2 + traj(3,:,:).^2)); 

% Use your key radius (pre-defined) to cutoff key vs keyhole data:
keyhole_rad = r_key*max(traj_rad(:)); 

% Define number of projections for each key:
narms_key1 = ceil(N_spirals/2);
narms_key2 = ceil(N_spirals/2);

%% 10) Normalize the k-space data:
% Goal: normalize the full image data so that all k0 values start at the
% same point (per slice). This mitigates the effects of HP Xe decay between
% projections, and thus reduces random edge artifacts.

% We also want to normalize the k0 data within each key image. In this
% case, we scale the k0 data for each spiral to start from the mean(k0) of
% that key. 

% Normalize all data to mean k0 for each spiral: 
k0(:) = abs(KSpaceDecay(1,:));
meank0_1 = mean(k0([1:narms_key1]));
meank0_2 = mean(k0([narms_key2 : end]));

for i = 1:N_spirals 
    for j = 1:N_samp 
        % Normalize the data for the full image:
        KSpaceDecay1(j,i) = KSpaceDecay(j,i).*meank0_1./k0(1,i);
        KSpaceDecay2(j,i) = KSpaceDecay(j,i).*meank0_2./k0(1,i);
        
        % Send data to key 1:
        Key1(j,i) = KSpaceDecay1(j,i);
        
        % Send data to key 2:
        Key2(j,i) = KSpaceDecay2(j,i);
    end
end

% Remove data values inside key: 
Key1(traj_rad < keyhole_rad) = nan;
Key2(traj_rad < keyhole_rad) = nan;

% Refill the key data with normalized k-space data:
% for i = 1:floor(N_spirals/2)
%     % Key 1 gets first half of spirals:
%     Key1(find(traj_rad(:,i)<keyhole_rad),i) = KSpaceDecay1(...
%         find(traj_rad(:,i)<keyhole_rad),i);
%     % Key 2 gets second half of spirals:
%     Key2(find(traj_rad(:,i+floor(size(traj,3)/2))<keyhole_rad),...
%         i+floor(size(traj,3)/2)) = KSpaceDecay2(...
%         find(traj_rad(:,i+floor(size(traj,3)/2))<keyhole_rad),...
%         i+floor(size(traj,3)/2));
% end

for i = 1:narms_key1  % Number of spirals in first key
    for j = 1:N_samp
%         Fill key data in first key with the actual data:
        if traj_rad(j,i) < keyhole_rad
            Key1(j,i) = KSpaceDecay1(j,i);
%             Key1(j,i + narms_key1) = KSpaceDecay1(j,i);
        end
    end
end
for i =  narms_key2 : N_spirals % Number of spirals in second key
    for j = 1:N_samp
%         Fill key data in second key with the actual data:
        if traj_rad(j,i) < keyhole_rad
            Key2(j,i) = KSpaceDecay2(j,i);
%             Key2(j,i - narms_key1 + 1) = KSpaceDecay2(j,i);
        end
    end
end


%% 11) Reshape trajectories and visualize the keyhole segregation:
% Reshape the trajectories to make an array that works for our recon.
trajx = reshape(traj(1,:,:),1,[])';
trajy = reshape(traj(2,:,:),1,[])';
trajz = reshape(traj(3,:,:),1,[])';
trajC = [trajx trajy trajz];

% Visualize:
f4 = figure('Name','Trajectories')
[traj_key1,traj_key2,keyhole1,keyhole2] = Functions.keyholetrajectories(...
    traj,traj_rad,keyhole_rad,0);
view(2)
axis square
axis off
title Spirals
pause(1)
% close(f4)

%% 12) Reconstruct the images using Scott's Reconstruction:
% For each key image, we will create a FID vector containing all the
% k-space information. If a k-space coordinate lies outside a respective
% key & keyhole, we will remove it.

% Key 1 image:
fid1 = reshape(Key1,1,[])';
keytraj = trajC;
zerofid = isnan(fid1);
fid1(zerofid) = [];
% fid1(:) = 1;
keytraj(zerofid,:) = [];
KeyImage1 = abs(rot90(+Functions.ScottRecon3D_Healthy_optimized(ImSize,fid1,keytraj),2));

% Key 2 image:
fid2 = reshape(Key2,1,[])';
keytraj = trajC;
zerofid = isnan(fid2);
fid2(zerofid) = [];
% fid2(:) = 1;
keytraj(zerofid,:) = [];
KeyImage2 = abs(rot90(+Functions.ScottRecon3D_Healthy_optimized(ImSize,fid2,keytraj),2));

% Full image:
fid_full = reshape(KSpaceDecay,1,[])';
traj_full = trajC;
zerofid = isnan(fid_full);
fid_full(zerofid) = [];
% fid_full(:) = 1;
traj_full(zerofid,:) = [];
ImageFull = abs(rot90(+Functions.ScottRecon3D_Healthy_optimized(ImSize,fid_full,traj_full),2));

% Re-orientate images to deal with the reconstruction output orientation:
KeyImage1 = flip(KeyImage1,1);
KeyImage2 = flip(KeyImage2,1);
KeyImage1 = flip(KeyImage1,2);
KeyImage2 = flip(KeyImage2,2);
ImageFull = flip(ImageFull,1);
ImageFull = flip(ImageFull,2);

% Store all three images in one full array.
MImageDK = zeros(size(KeyImage1,1),size(KeyImage1,2),3);
MImageDK(:,:,1) = KeyImage1;
MImageDK(:,:,2) = KeyImage2;
MImageDK(:,:,3) = ImageFull;

%% 13) Calculate flip angle map:
% Attenuation map:
Attenuation = KeyImage2./KeyImage1;

% Calculated FA map:
FA_Calc = abs(acosd(nthroot(Attenuation, narms_key2)).*Mask);
FA_Calc(isinf(FA_Calc)|isnan(FA_Calc)) = 0;

% Resize the applied flip angle map: (sadly, this may cause blurring)
FAMap = imresize(FAMap,[ImageSize ImageSize]);

%% 14) Correct the image:
% This section is controversial. Before publishing, ensure we have agreed
% upon a 'standard' correction method (i.e. inclusion of alpha^(-1) yes/no)
CorMap = (narms_key2./sind(FA_Calc)).*((1-cosd(FA_Calc))./(1-(cosd(FA_Calc).^(narms_key2))));
CorImage = (ImageFull.*CorMap).*Mask;
CorImage(isinf(CorImage)|isnan(CorImage)) = 0;

%% 15) Visualize results:
% Set colormap limits:
CMap = jet;
CMap(1,:) = [0 0 0];

% Normalize to 99.9th percentile to remove extra bright artifacts:
MImageDK = MImageDK./prctile(MImageDK,99,'all');
CorImage = CorImage./prctile(CorImage,99,'all'); 

% Generate a figure:
summary = figure('Name','Keyhole Reconstruction Summary','units','normalized','outerposition',[0 0 1 1])

% Generate text window to show the parameters that went into this
% simulation:
scan_str = ["Projections: " + N_spirals;...
    "Dwell time: " + dwell + "ms";...
    "Angle type: " + beta_correction;...
    "Ordering type: " + ordering_type;...
    "Flip angles: " + FA(1)+"," + FA(2)+"," + FA(3)+"," + FA(4);...
    "Flip angle pattern: " + pattern;...
    "Overgridding factor: " + OverSize]
recon_str = ["Number of keys: " + N_keys;...
    "Key radius: " + 100*round((r_key),3) + "%";...
    "Undersampling factor: " + round(R,2)];
ax = subplot(3,3,1);
text(0,0.7,scan_str);
text(0,0.1,recon_str,'FontWeight','Bold');
title('Simulation Settings')
axis off

% Original image:
subplot(3,3,2)
imagesc(MySlice)
axis square
axis off
colormap(gca, gray); colorbar;
caxis([0 1])
title('Original Image')

% Applied flip angle map:
subplot(3,3,3)
imagesc(Mask.*FAMap)
axis square
axis off
colormap(gca,CMap)
caxis(gca,[0 max(FA)+10]); colorbar;
title('Applied Flip Angle')

% Key 1 image:
subplot(3,3,4)
imagesc(MImageDK(:,:,1).*Mask)
axis square
axis off
colormap(gca, gray); 
caxis([0 1]); 
colorbar;
title('Key Image 1')

% Key 2 image:
subplot(3,3,5)
imagesc(MImageDK(:,:,2).*Mask)
axis square
axis off
colormap(gca, gray); 
caxis([0 1]); 
colorbar;
title('Key Image 2')

% Full image:
subplot(3,3,5)
imagesc(MImageDK(:,:,3).*Mask)
axis square
axis off
colormap(gca, gray); 
caxis([0 1]); 
colorbar;
title('Full image')

% Calculated/measured flip angle:
subplot(3,3,6)
imagesc(FA_Calc)
axis square
axis off
colormap(gca,CMap)
caxis(gca,[0 max(FA)+10]); colorbar;
title('Measured Flip Angle')
hold off

% Overview of k-space decay for all data:
subplot(3,3,7)
x1 = [1 : length(fid1)];
x2 = [length(fid1) + 1 : length(fid1) + length(fid2)];
plot(x1, abs(fid1),'.r')
hold on
plot(x2, abs(fid2),'.k')
xlabel('Sampling number')
ylabel('k-space data value')
legend('Image 1 Data','Image 2 Data')
title('FID Visualization')

% Keyhole data segregation:
subplot(3,3,8)
Functions.keyholetrajectories(traj,traj_rad,keyhole_rad,0);
view(2)
axis square

% Bland Altman plot:
subplot(3,3,9)
Data1 = (Mask.*FAMap);
Data2 = (Mask.*FA_Calc);
Data1 = nonzeros(Data1(:))';
Data2 = nonzeros(Data2(:))';
[~,~,meandiff, CR,~] = Functions.BlandAltman(Data1, Data2, 3)
xlabel('Mean Flip Angle ((App+Meas)/2)')
ylabel('Difference Flip Angle (App-Meas)')
hold on
plot(FA, zeros(size(FA,1)),'ko',...
    'MarkerFaceColor',[.49 1 .63],...
    'MarkerSize',5)
ylim([-15 15])
title('Bland Altman Plot',['Mean Diff = ',num2str(meandiff),', Sigma = ',num2str((CR(1)-CR(2))/3.98)]);
RMSD = sqrt(mean((Data1 - Data2).^2)) % Root mean square difference
RSSD = sqrt(sum((Data1 - Data2).^2)) % Root sum square difference (A.K.A. L2-norm)


%% 16) Visualize final results concisively:
figure('Name','Final flip angle maps')
% Applied flip angle map:
subplot(1,2,1)
imagesc(Mask.*FAMap)
axis square
axis off
colormap(gca,CMap)
caxis(gca,[0 max(FA)+10]); 
% colorbar;
% title('Applied Flip Angle')

% Calculated/measured flip angle:
subplot(1,2,2)
imagesc(FA_Calc)
axis square
axis off
colormap(gca,CMap)
caxis(gca,[0 max(FA)+10]); 
% colorbar;
% title('Measured Flip Angle')
hold off

%% 17) Visualize full and corrected images: 
% Generate figure:
correction = figure('Name','Bias Field Correction Summary')

% Full image:
subplot(2,1,1)
imagesc(MImageDK(:,:,3).*Mask)
axis square
axis off
colormap(gca, gray); 
colorbar;
caxis([0 1])
title('Full uncorrected Image')

% Corrected image:
subplot(2,1,2)
imagesc(CorImage.*Mask)
axis square
axis off
colormap(gca, gray); 
colorbar;
caxis([0 1])
title('Corrected Image')

%% 18) Save data:
% Save figure:
% filename = [ + ".png"];
% saveas(summary,['  ' + filename])

% Save workspace:
% save('LOCATION\NAME.mat')
% 
% %% 19) Uncertainty calculations:
% % The agenda of this section is to determine the uncertainty (analytical
% % error) in our flip angle calculations as a function of signal in each
% % voxel. For more information on derivations, contact Mariah Costa.
% 
% % Calculate C1:
% C1_Calc = abs(nthroot(Attenuation, narms_key2).*Mask);
% C1_Calc(isinf(C1_Calc)|isnan(C1_Calc)) = 0;
% 
% % Calculate uncertainty in the signal, ie the standard dev. of the noise:
% count =0;
% for i =  1:size(ImageFull,1)
%     for j =  1:size(ImageFull,2)
%        if Mask(i,j) ==0
%            count = count + 1;
%            FullIm_Noise(count) = ImageFull(j,j);
%        end
%     end
% end
% dS = std(FullIm_Noise);
% 
% % Calculate uncertainty per pixel:
% for i = 1:size(KeyImage1,1)
%     for j = 1:size(KeyImage1,2)
%         if Mask(i,j) == 1
%             FA_Unc(i,j) = (2*dS/N_spirals).*(C1_Calc(i,j)./KeyImage1(i,j)).*sqrt( 1./(1-C1_Calc(i,j)).*(1+(1./(C1_Calc(i,j).^(N_spirals)))));
%         else
%             FA_Unc(i,j) = 0;
%         end
%     end
% end
% 
% % Find relative uncertainty percentage:
% FA_RelUnc = FA_Unc./FAMap.*100;
% 
% % Find Top fence for outliers & use this fence for caxis colorscale - MC TAG
% Range_FA_RelUnc = FA_RelUnc;
% Range_FA_RelUnc(Range_FA_RelUnc==0)=[];
% IQR = iqr(Range_FA_RelUnc);
% ThirdQuartile = quantile(Range_FA_RelUnc,0.75,'all');
% TopFence = ThirdQuartile + 3*IQR;
% 
% figure;
% imagesc(FA_RelUnc); 
% colormap(gca,jet)
% caxis(gca,[0 TopFence]); colorbar;
% title('Percent Uncertainty in Flip Angle Calculation')
%  


