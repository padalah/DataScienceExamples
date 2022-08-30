# Databricks notebook source
#####
# Goal: To predict airplane delays at least 2 hours before they happend
# What is a delay? Arrival or departure of an aircraft happens 15 mins after scheduled time
# Factors affecting delays:
#   - weather
#   - patrons
#   - other flight delays
# Possible ways to predict delays:
#   Binary: Will a flight be delayed?
#   Model each flights duration and use regression to identify flights predicted duration. Compare with expected duration
#   Look at frequency to see if certain flights are delayed more often
#   Using graph algorithms focus on a subset of to-from locations and optimize shortest path for fligths that are most likely to be delayed.
#   Another option is to use probability based networks but I am not quite sure how this would work
