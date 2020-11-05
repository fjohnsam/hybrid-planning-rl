(define(domain lunarlander)
(:requirements :strips :equality :typing :adl :fluents )
(:types position velocity destination)
(:predicates
    (reached ?d - destination)
)
(:functions
    (xpos ?px - position)
    (ypos ?py - position)
    (xvel ?vx - velocity)
    (yvel ?vy - velocity)
    (reward)
)

(:action stop
    :parameters(?px ?py - position ?vx ?vy - velocity ?d - destination)
    :precondition(and
                (not (reached ?d))
                ; (<= (xpos ?px) 10)
                (= (xpos ?px) 0)
                ; (<= (ypos ?py) 10)
                (= (ypos ?py) 0)
                (= (xvel ?vx) 0)
                ;(>= (xvel ?vx) -20)
                (= (yvel ?vy) 0)
                ;(>= (yvel ?vy) -20)
    )
    :effect(and
            (reached ?d)
            (increase(reward) 200)
    )
)
(:action m1s1
        :parameters(?px ?py - position ?vx ?vy - velocity ?d - destination)
        :precondition (and
                (not (reached ?d))
        )
        :effect(and
                (increase (xpos ?px) (xvel ?vx))
                (increase (ypos ?py) (yvel ?vy))
                (increase (xvel ?vx) 1)
                (increase (yvel ?vy) 1)
                (decrease (reward) 3)
        )
)
(:action m-1s-1
        :parameters(?px ?py - position ?vx ?vy - velocity ?d - destination)
        :precondition (and
                (not (reached ?d))
        )
        :effect(and
                (increase (xpos ?px) (xvel ?vx))
                (increase (ypos ?py) (yvel ?vy))
                (decrease (xvel ?vx) 1)
                (decrease (yvel ?vy) 1)
                (decrease (reward) 3)
        )
)
(:action m1s-1
        :parameters(?px ?py - position ?vx ?vy - velocity ?d - destination)
        :precondition (and
                (not (reached ?d))
        )
        :effect(and
                (increase (xpos ?px) (xvel ?vx))
                (increase (ypos ?py) (yvel ?vy))
                (increase (xvel ?vx) 1)
                (decrease (yvel ?vy) 1)
                (decrease (reward) 3)
        )
)
(:action m-1s1
        :parameters(?px ?py - position ?vx ?vy - velocity ?d - destination)
        :precondition (and
                (not (reached ?d))
        )
        :effect(and
                (increase (xpos ?px) (xvel ?vx))
                (increase (ypos ?py) (yvel ?vy))
                (decrease (xvel ?vx) 1)
                (increase (yvel ?vy) 1)
                (decrease (reward) 3)
        )
)
)