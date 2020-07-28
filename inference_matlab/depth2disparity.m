function [ disprity ] = depth2disparity( depth, Baseline, f )

disprity = Baseline*f ./ depth;
idx= disprity==inf;
disprity(idx) = 0;

end

