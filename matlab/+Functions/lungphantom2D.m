function im = lungphantom2D(ImSize)
cent1x = 5*ImSize/16;
cent2x = 11*ImSize/16;
centy = ImSize/2;

[X,Y] = meshgrid(1:ImSize,1:ImSize);

%Make 2 elipses to be lungs - we'll make one have sharply varying signal
%intensities, and one to have smoothly varying signal intensities
el1 = (X-cent1x).^2/(3*ImSize/16)^2 + (Y-centy).^2/(6*ImSize/16)^2;
% el1 = (X-cent1x).^2/(3*ImSize/32)^2 + (Y-centy).^2/(6*ImSize/32)^2;
% el1 = (X-cent1x).^2/(4*ImSize/16)^2 + (Y-centy).^2/(6*ImSize/16)^2;

el1(el1<=1) = 1;
el1(el1~=1) = 0;

el2 = (X-cent2x).^2/(3*ImSize/16)^2 + (Y-centy).^2/(6*ImSize/16)^2;
% el2 = (X-cent2x).^2/(3*ImSize/32)^2 + (Y-centy).^2/(6*ImSize/32)^2;
% el2 = (X-cent2x).^2/(4*ImSize/16)^2 + (Y-centy).^2/(6*ImSize/16)^2;

el2(el2<=1) = 1;
el2(el2~=1) = 0;

%Make left lung sharply varying:
sharp = ones(ImSize,ImSize);
sharp(find(X<cent1x)) = 0.5;
sharp(Y<ImSize/4) = sharp(Y<ImSize/4)*0.4;
sharp(Y>=ImSize/4 & Y<ImSize/2) = sharp(Y>=ImSize/4 & Y<ImSize/2)*0.6;
sharp(Y>=ImSize/2 & Y<3*ImSize/4) = sharp(Y>=ImSize/2 & Y<3*ImSize/4)*0.8;

el1 = el1.*sharp;

%Make right lung smoothly varying:
sigx = ImSize/8;
sigy = ImSize/4;
% smooth = exp(-((X-cent2x).^2/(2*sigx^2) + (Y-centy).^2/(2*sigy^2)));
smooth = exp(-((X-cent2x).^2/(4*sigx^2) + (Y-centy).^2/(2*sigy^2))); % More signal on sides
%figure;imagesc(smooth)
el2 = el2.*smooth;

im = el1 + el2;

% figure('Name','Double Check Phantom')
% imagesc(im)
% axis square
% axis off
% colormap gray