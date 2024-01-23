function [order, ordering_type] = philipsreorder(narms, ordering)
%%%% Date: %%%%
% 07/30/2021
%
%%%% Author information: %%%%
% Joseph Plummer
%
%%%% Agenda: %%%%
% A function to generate a keyhole animation for the 2-key 2D-spiral approach.
% This is designed to show the simultaneous spiral trajectories for each RF
% pulse.
%
% For reference: https://doi.org/10.1002/(sici)1522-2594(199902)41:2%3C417::aid-mrm29%3E3.0.co;2-w
% Pipe JG, Ahunbay E, Menon P. Effects of interleaf order for spiral MRI of
% dynamic processes. Magn Reson Med. 1999.
% 
% Inputs:
% narms = number of projections
% ordering = type of ordering style
%
% Outputs:
% order = array showing the desired order of projections
% ordering_type = name of ordering style
%
%% Script:
% Set defaults:
if nargin == 2
    narms = narms;
    ordering = ordering;
elseif nargin == 1;
    narms = narms;
    ordering = "Ascending" % Default
else nargin == 0;
    disp("Input error. Please enter number of projections, and the ordering strategy.")
    return
end

% Declare ordering strategy:
switch ordering
    
    case "Ascending"
        % Ascending is the default: (e.g. [1 2 3 4 5 ...])
        ordering_type = "Ascending (default)";
        for ii = 1:narms
            order(ii) = ii;
        end
        
    case "Descending"
        % Descending: (e.g. [30 29 28 27 ...])
        ordering_type = "Descending";
        for ii = 1:narms
            order(ii) = narms - ii + 1;
        end
        
    case "Two-way"
        % Two-way (ascending-descending): (e.g. [1 30 2 29 3 28 4 27 5 26 6 25 7 ...])
        ordering_type = "Two-way";
        order_asc = [1:narms];
        order_des = [narms:-1:1];
        for ii = 1:narms
            if mod(ii,2) == 1; % Odd index
                order(ii) = order_asc(ceil(ii/2));
            else mod(ii,2) == 0; % Even index
                order(ii) = order_des(ceil(ii/2));
            end
        end
        
    case "Skipped"
        % Skipped: (Evens, then odds)
        ordering_type = "Skipped";
        j = 1;
        for ii = 1:narms
            if mod(ii,2) == 0; % Even index
                order(j) = ii;
                j = j + 1;
            else
            end
        end
        for ii = 1:narms
            if mod(ii,2) == 1; % Odd index
                order(j) = ii;
                j = j + 1;
            else
            end
        end
        
    case "Mixed"
        % Combination of both Two-way and Skipped ordering schemes:
        ordering_type = "Mixed";
        % First, set up Skipped:
        j_even = 1;
        j_odd = 1;
        for ii = 1:narms
            if mod(ii,2) == 0; % Even
                evens_asc(j_even) = ii;
                evens_des = flip(evens_asc);
                j_even = j_even + 1;
            else mod(ii,2) == 1; % Odd
                odds_asc(j_odd) = ii;
                odds_des = flip(odds_asc);
                j_odd = j_odd + 1;
            end
        end
        
        % Second, take Two-way ordering of both the evens and odds:
        for ii = 1:length(evens_asc)
            if mod(ii,2) == 1; % Odd index
                tw_evens(ii) = evens_asc(ceil(ii/2));
            else mod(ii,2) == 0; % Even index
                tw_evens(ii) = evens_des(ceil(ii/2));
            end
        end
        
        for ii = 1:length(odds_asc)
            if mod(ii,2) == 1; % Odd index
                tw_odds(ii) = odds_asc(ceil(ii/2));
            else mod(ii,2) == 0; % Even index
                tw_odds(ii) = odds_des(ceil(ii/2));
            end
        end
        
        % Finally, merge the two ordering regimes together. The strategy is
        % different if there is an even number of spirals vs an odd number of
        % spirals.
        if mod(narms,2) == 0 && mod(narms/2,2) == 1; % Even number of spirals, 2x odd number of spirals
            for ii = 1:narms/2
                if mod(ii,2) == 1; % Odd index
                    order(ii) = tw_evens(ii);
                    order(ii + narms/2) = tw_odds(ii);
                else mod(ii,2) == 0; % Even index
                    order(ii) = tw_odds(ii);
                    order(ii + narms/2) = tw_evens(ii);
                end
            end
        elseif mod(narms,2) == 0 && mod(narms/2,2) == 0; % Even number of spirals, 2x even number of spirals
            for ii = 1:narms/2
                if mod(ii,2) == 1; % Odd index
                    order(ii) = tw_evens(ii);
                    order(ii + narms/2) = tw_odds(ii);
                else mod(ii,2) == 0; % Even index
                    order(ii) = tw_odds(ii);
                    order(ii + narms/2) = tw_evens(ii);
                end
            end
            % Correct the array using circshift:
            order([narms/2:narms]) = circshift(order([narms/2:narms]),-1);
        else mod(narms,2) == 1; % Odd number of spirals
            order = [tw_evens, tw_odds];
        end

    otherwise
        disp("Input error. Please enter an ordering strategy from: Ascending, Descending, Two-way, Skipped, Mixed.")
        return
   
end
