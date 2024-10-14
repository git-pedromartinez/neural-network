/**
 * Author: Pedro Martinez
 * Email: id.pedromartinez@gmail.com
 * Position: Senior Software Engineer
 */

export type MatrixType = "1D" | "2D" | "3D";

export type Matrix<D extends MatrixType = "1D", T = number> = D extends "1D"
                                                            ? T[]
                                                            : D extends "2D"
                                                            ? T[][]
                                                            : D extends "3D"
                                                            ? T[][][]
                                                            : never;

// Example of using the Matrix interface

const arr: Matrix = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
const arr1: Matrix<"1D"> = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
const arr2: Matrix<"2D"> = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]];
const arr3: Matrix<"3D"> = [[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]];