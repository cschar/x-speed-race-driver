
-- activate application "Chrome"
-- repeat 5 times
--     -- tell application "System Events" to keystroke 126 
--     tell application "System Events"
--     		key code 126
--     end
--     -- tell application "System Events" to keystroke 30 using command down
--     delay (random number from 0.5 to 2)
--     -- tell application "System Events" to keystroke "v" using command down
-- end repeat

activate application "XSpeedRace2"
tell application "System Events"
        -- works!
        repeat 30 times
            key code {}
            delay 0.01  
        end
end