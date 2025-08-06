function solution (X,Y,D){
    const distance = Y - X;

    jump = Math.ceil(distance/D)

    return jump
}