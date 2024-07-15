// Throws an exception if the conditional is false.
export const assert = (cond : boolean, text : string) => {
    if (cond === false) { throw Error(text) }
}

// Throws an exception if the switch case being matched on is inexhaustive. It should give an error at compile time first.
export const assert_tag_is_never = (tag : never): never => { throw new Error(`Invalid tag. Got: ${tag}`)};
export const min = (a : number,b : number) => a < b ? a : b;
export const max = (a : number,b : number) => a >= b ? a : b;
export const clamp = (x : number, _min : number, _max : number) => 
    _min <= _max && !isNaN(x) && !isNaN(_min) && !isNaN(_max) 
    ? min(max(_min,x), _max)
    : (() => {throw Error(`Invalid args in clamp. Got: ${[x,_min,_max]}`)})();