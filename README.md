# Functions
  ## simulate_spring() 
    function that simulate the collisions for a given N value, friction, speed, and elasticity
  ## collect_data()
    runs simulate_spring() for a range of N values, and outputs the results as both termial text and as CVS file
  ## animate()
    uses info from simulate_spring() to create little graphics of what it would look like 

# Running it
  if you don't want to just use the defaults for all the functions, you will have to set what you want to when you call the functions
  
  the default max for the highest N value you can call with collect_data is 10, so if you want to go higher, you will have to set the max highter when you call collect_data 
  
  there isn't a "make a bunch of animations function" so you will have to use a "for n in range()" or something like that
