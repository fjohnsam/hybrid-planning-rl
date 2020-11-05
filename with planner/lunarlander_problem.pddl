(define (problem ll)
(:domain lunarlander)
(:objects
        landerpx - position
        landerpy - position
        landervx - velocity
        landervy - velocity
        d - destination
)
(:init 
        (= (xpos landerpx) -5)
        (= (ypos landerpy) 95)
        (= (xvel landervx) 0)
        (= (yvel landervy) 0)
        (not(reached d))
        (= (reward) 100)
)
(:goal (reached d))

(:metric maximize (reward))
)
;(:goal (and
;       (reached d)
;))