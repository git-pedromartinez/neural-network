/**
 * Author: Pedro Martinez
 * Email: id.pedromartinez@gmail.com
 * Position: Senior Software Engineer
 */

/**
 * @description This function returns the total execution time, by default in seconds
 *
 * @param startTime
 * @returns
 */
export function totalTime(startTime: Date): number {
  return (new Date().getTime() - startTime.getTime()) / 1000;
}
