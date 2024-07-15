// Throws an exception if the conditional is false.
export const assert = (cond : boolean, text : string) => {
    if (cond === false) { throw Error(text) }
}