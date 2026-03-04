"use client";
import { cn } from "@/lib/utils";
import React, { useEffect, useRef, useState } from "react";
import { createNoise3D } from "simplex-noise";

const get_point = (magnitude : number, theta : number) => {
  return {
    x: magnitude * Math.cos(theta),
    y: magnitude * Math.sin(theta),
  };
}


export const QueryLakeLogo = ({
  children,
  className,
  containerClassName,
  canvasClassName,
  colors,
  waveWidth,
  backgroundFill,
  blur = 10,
  speed = 5,
  waveOpacity = 0.5,
  waveCount = 5,
  waveAmplitude = 100,
  wavePinchMiddle = 0,
  wavePinchEnd = 0,
  ...props
}: {
  children?: any;
  className?: string;
  containerClassName?: string;
  canvasClassName?: string;
  colors?: string[];
  waveWidth?: number;
  backgroundFill?: string;
  blur?: number;
  speed?: number;
  waveOpacity?: number;
  waveCount?: number;
  waveAmplitude?: number;
  wavePinchMiddle?: number;
  wavePinchEnd?: number;
  [key: string]: any;
}) => {
  const noise = createNoise3D();
  const start_time = Date.now();
  let w: number,
      h: number,
      nt: number,
      i: number,
      x: number,
      ctx: any,
      canvas: any,
      animation_points: number = 40,
      time_passed: number = 0,
      interval : number = 0;
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const speedSet = Array.from({length: waveCount}, () => speed * (Math.random() * 0.9 + 0.1));

  const init = () => {
    canvas = canvasRef.current;
    ctx = canvas.getContext("2d");

    let parent = canvas.parentElement;

    // Set the canvas dimensions to match the parent's dimensions
    if (parent) {
      canvas.width = parent.offsetWidth;
      canvas.height = parent.offsetHeight;
      w = ctx.canvas.width = canvas.width;
      h = ctx.canvas.height = canvas.height;
      interval = w / animation_points;
    }

    // ctx.filter = `blur(${blur}px)`;
    nt = 0;
    window.onresize = function () {
      canvas.width = parent.offsetWidth;
      canvas.height = parent.offsetHeight;
      w = ctx.canvas.width = canvas.width;
      h = ctx.canvas.height = canvas.height;
      // w = ctx.canvas.width = window.innerWidth;
      // h = ctx.canvas.height = window.innerHeight;
      interval = w / animation_points;
      // ctx.filter = `blur(${blur}px)`;
    };
    render();
  };

  const waveColors = colors ?? [
    "#38bdf8",
    "#818cf8",
    "#c084fc",
    "#e879f9",
    "#22d3ee",
  ];
  
  // const drawWave = (n: number) => {
  //   time_passed = Date.now() - start_time;
  //   for (i = 0; i < n; i++) {
  //     const lt = speedSet[i] * 0.001 * time_passed * 0.05;
  //     ctx.beginPath();
  //     ctx.lineWidth = waveWidth || 50;
  //     ctx.strokeStyle = waveColors[i % waveColors.length];
  //     for (x = 0; x < w; x += interval) {
  //       const xNormalized = ( x ) / ( w );
  //       const pinchedAmplitude = waveAmplitude * ( (1 - wavePinch) * Math.sin( Math.PI * xNormalized) + wavePinch );

  //       var y = noise(x / 800, 2 * i, lt) * pinchedAmplitude;
  //       ctx.lineTo(x, y + h * 0.5); // adjust for height, currently at 50% of the container
  //     }
  //     ctx.stroke();
  //     ctx.closePath();
  //   }
  // };

  const drawWave = (n: number) => {
    time_passed = Date.now() - start_time;
    let points : number[][][] = [];
    const calc_callbacks = [(n_2 : number) => get_point(waveAmplitude, Math.PI*(2*n_2/n + 1/n))]
    const lt = 0.001 * time_passed * 0.05;
    const R_COUNT = 5

    for (let r = 0; r < R_COUNT; r++){
      ctx.lineWidth = 10;
      ctx.strokeStyle = waveColors[r % waveColors.length];
      if (r === 1) {
        calc_callbacks.push((n_2 : number) => {
          return get_point(waveAmplitude*Math.cos(Math.PI/n), Math.PI*(2*n_2/n));
        })
      } else if (r > 1) {
        calc_callbacks.push((n_2 : number) => {
          const p_1 = calc_callbacks[r-1](n_2),
                p_2 = calc_callbacks[r-1](n_2 + 1),
                p_3 = calc_callbacks[r-2](n_2 + ((r > 2)?1:0));
          return {x: (p_1.x + p_2.x - p_3.x), y: (p_1.y + p_2.y - p_3.y)};
        })
      }
      const sample_pont = calc_callbacks[r](0);
      const magnitude = Math.sqrt(sample_pont.x**2 + sample_pont.y**2);

      let current_points : number[][] = [];
      for (let i = 0; i < n; i++) {
        const point = get_point(
          magnitude * Math.min(w,h)*0.5, 
          Math.PI*(2*i/n + ((r % 2 === 0)?0:1/n) + 1/n)
        );
        // ctx.beginPath();
        // ctx.arc(point.x + w * 0.5, point.y + h * 0.5, 1, 0, 2 * Math.PI);
        // ctx.stroke();

        let new_point = {x: point.x, y: point.y};
        // var noise_make = noise(new_point.x / 20, new_point.y / 20, lt * 20);
        // new_point.x *= 1 - noise_make * 0.01;
        // new_point.y *= 1 - noise_make * 0.01;

        current_points.push([new_point.x + w * 0.5, new_point.y + h * 0.5]);
      }
      points.push(current_points);
      ctx.lineWidth = 2;
      // ctx.strokeStyle = waveColors[r % waveColors.length];
      ctx.strokeStyle = "#FFFFFF";
      // console.log("points", points.length)
      if (r > 0) {

        for (let i = 0; i < n; i++) {
          const wrapped_i = (i+((r%2===0)?(-1):(1)));
          const next_point_i = (wrapped_i < 0)?((wrapped_i + n) % n):(wrapped_i % n);
          ctx.beginPath();
          ctx.moveTo(points[r][i][0], points[r][i][1]);
          // ctx.lineTo(points[r][next_point_i][0], points[r][next_point_i][1]);
          // ctx.lineTo(points[r-1][next_point_i][0], points[r-1][next_point_i][1]);
          // ctx.lineTo(points[r][i][0], points[r][i][1]);
          if (r === R_COUNT - 1) {
            ctx.lineTo(points[r][next_point_i][0], points[r][next_point_i][1]);
            ctx.lineTo(points[r-1][next_point_i][0], points[r-1][next_point_i][1]);
            ctx.lineTo(points[r][i][0], points[r][i][1]);
          } else {
            ctx.lineTo(points[r-1][next_point_i][0], points[r-1][next_point_i][1]);
            // ctx.lineTo(points[r][i][0], points[r][i][1]);
            ctx.lineTo(points[r][i][0], points[r][i][1]);
            ctx.lineTo(points[r-1][i][0], points[r-1][i][1]);
          }

          // const next_point = points[r-1][next_point_i];
          // console.log("next_point", next_point);
          // ctx.moveTo(points[r][i][0], points[r][i][1]);
          // ctx.lineTo(next_point[0], next_point[1]);
          // ctx.lineTo(points[r][i][0], points[r][i][1]);
          
          ctx.closePath();
          // if (r === 7) {

          //   // if (i % 2 === 1)
          //   ctx.fillStyle = "#FFFFFF";
          //   ctx.fill();
          // }
          // console.log("next_point", next_point)
          // ctx.lineTo(points[r-1][(i-1)%points[r-1].length][0], points[r-1][(i-1)%points[r-1].length][1]);
          // ctx.lineTo(points[r-1][(i-1)%n][0], points[r-1][(i-1)%n][1]);
          ctx.stroke();
        }
      }

    }

    // for (i = 0; i < n; i++) {
    //   const lt = speedSet[i] * 0.001 * time_passed * 0.05;
    //   ctx.beginPath();
    //   ctx.lineWidth = waveWidth || 50;
    //   ctx.strokeStyle = waveColors[i % waveColors.length];
      
      
    //   // Define the control points for the Bezier curve
    //   const controlPoints = [];
    //   const canvasHeight = canvasRef.current?.height || 100;
    //   for (x = 0; x < w; x += interval) {
    //     const xNormalized = (x) / (w); // x normalized to [0, 1]
    //     const expControl = Math.exp(-Math.pow(50 * (xNormalized - 0.5) * wavePinchMiddle, 2));
    //     const pinchedAmplitude = waveAmplitude * canvasHeight * ((1 - wavePinchEnd) * (1 + Math.sin(Math.PI * xNormalized) * expControl + wavePinchEnd));

    //     var y = noise(x, 2 * i, lt) * waveAmplitude * canvasHeight * 0.5 + Math.min(w/2, h/2) - 100;
        
    //     // var y = noise(x / 800, 2 * i, lt) * waveAmplitude * canvasHeight + 2;
    //     // controlPoints.push({ x, y: y + h * 0.5 });

    //     var x_from_polar = Math.cos(xNormalized * 2 * Math.PI) * y + w / 2;
    //     var y_from_polar = Math.sin(xNormalized * 2 * Math.PI) * y + h / 2;
    //     controlPoints.push({ x: x_from_polar, y: y_from_polar });
    //   }

    //   // Add a single point at the end of w
    //   const xEnd = w;
    //   const xNormalizedEnd = xEnd / w;
    //   const pinchedAmplitudeEnd = waveAmplitude * ((1 - wavePinchEnd) * Math.sin(Math.PI * xNormalizedEnd) + wavePinchEnd);
    //   const yEnd = noise(xEnd / 800, 2 * i, lt) * pinchedAmplitudeEnd;
    //   // controlPoints.push({ x: xEnd, y: yEnd + h * 0.5 });
    //   // controlPoints.push({ x: w, y: h / 2 });
    //   controlPoints.push(controlPoints[0]);
    //   controlPoints.push(controlPoints[0]);
  
    //   // Draw the Bezier curve
    //   ctx.moveTo(controlPoints[0].x, controlPoints[0].y);
    //   let j;
    //   for (j = 0; j < controlPoints.length - 2; j++) {
    //     ctx.strokeStyle = `rgba(${waveColors[i % waveColors.length]}, ${1 - j / blur})`;
    //     const xc = (controlPoints[j].x + controlPoints[j + 1].x) / 2;
    //     const yc = (controlPoints[j].y + controlPoints[j + 1].y) / 2;
    //     ctx.quadraticCurveTo(controlPoints[j].x, controlPoints[j].y, xc, yc);
    //   }
    //   // curve through the last two points
    //   ctx.quadraticCurveTo(controlPoints[j].x, controlPoints[j].y, controlPoints[j+1].x,controlPoints[j+1].y);
    //   ctx.stroke();
    //   ctx.closePath();
    // }
  };

  let animationId: number;
  const FPS = 60; // Set a target frame rate (e.g., 30 FPS)
  let lastRenderTime = Date.now();

  const render = () => {
    const currentTime = Date.now();
    const timeSinceLastRender = currentTime - lastRenderTime;

    if (timeSinceLastRender > 1000 / FPS) { // Only render if enough time has passed
      ctx.fillStyle = backgroundFill || "white";
      ctx.globalAlpha = waveOpacity || 0.5;
      ctx.fillRect(0, 0, w, h);
      drawWave(waveCount);
      lastRenderTime = currentTime;
    }
    // ctx.drawImage();

    animationId = requestAnimationFrame(render);
  };

  useEffect(() => {
    init();
    return () => {
      cancelAnimationFrame(animationId);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const [isSafari, setIsSafari] = useState(false);
  useEffect(() => {
    // I'm sorry but i have got to support it on safari.
    setIsSafari(
      typeof window !== "undefined" &&
        navigator.userAgent.includes("Safari") &&
        !navigator.userAgent.includes("Chrome")
    );
  }, []);

  return (
    <div
      className={cn(
        "rounded-md bg-neutral-950 relative flex flex-col items-center justify-center antialiased",
        containerClassName
      )}
    >
      <canvas
        className={cn(
          "transform-gpu absolute inset-0 z-0 bg-inherit h-inherit w-screen",
          // `blur-[${blur}px]`,
          canvasClassName
        )}
        ref={canvasRef}
        id="canvas"
        style={{
          willChange: 'transform', // Hint the browser for GPU acceleration
          // ...(isSafari ? { filter: `blur(${blur}px)` } : {}),
        }}
      ></canvas>
      <div className={cn("relative z-10", className)} {...props}>
        {children}
      </div>
    </div>
  );
};


export const create_logo_svg = () => {
  // time_passed = Date.now() - start_time;
  const n = 8;
  const r_count = 5;

  let points : number[][][] = [];
  const calc_callbacks = [(n_2 : number) => get_point(waveAmplitude, Math.PI*(2*n_2/n + 1/n))]
  // const lt = 0.001 * time_passed * 0.05;
  const waveAmplitude = 100, h = 600, w = 300;

  let paths : number[][] = [];
  let polygons : number[][] = [];

  for (let r = 0; r < r_count; r++){
    
    if (r === 1) {
      calc_callbacks.push((n_2 : number) => {
        return get_point(waveAmplitude*Math.cos(Math.PI/n), Math.PI*(2*n_2/n));
      })
    } else if (r > 1) {
      calc_callbacks.push((n_2 : number) => {
        const p_1 = calc_callbacks[r-1](n_2),
              p_2 = calc_callbacks[r-1](n_2 + 1),
              p_3 = calc_callbacks[r-2](n_2 + ((r > 2)?1:0));
        return {x: (p_1.x + p_2.x - p_3.x), y: (p_1.y + p_2.y - p_3.y)};
      })
    }
    const sample_pont = calc_callbacks[r](0);
    const magnitude = Math.sqrt(sample_pont.x**2 + sample_pont.y**2);

    let current_points : number[][] = [];
    for (let i = 0; i < n; i++) {
      const point = get_point(
        magnitude * Math.min(w,h)*0.5 / 100, 
        Math.PI*(2*i/n + ((r % 2 === 0)?0:1/n) + 1/n)
      );
      // ctx.beginPath();
      // ctx.arc(point.x + w * 0.5, point.y + h * 0.5, 1, 0, 2 * Math.PI);
      // ctx.stroke();

      let new_point = {x: point.x, y: point.y};
      // var noise_make = noise(new_point.x / 20, new_point.y / 20, lt * 20);
      // new_point.x *= 1 - noise_make * 0.01;
      // new_point.y *= 1 - noise_make * 0.01;

      current_points.push([new_point.x + w * 0.5, new_point.y + h * 0.5]);
    }
    points.push(current_points);
    // ctx.lineWidth = 2;
    // ctx.strokeStyle = waveColors[r % waveColors.length];
    // ctx.strokeStyle = "#FFFFFF";
    // console.log("points", points.length)
    if (r >= 2) {

      for (let i = 0; i < n; i++) {
        const wrapped_i = (i+((r%2===0)?(-1):(1)));
        const next_point_i = (wrapped_i < 0)?((wrapped_i + n) % n):(wrapped_i % n);
        if (r === r_count - 1) {
          paths.push([points[r][i][0], points[r][i][1], points[r][next_point_i][0], points[r][next_point_i][1]])
          paths.push([points[r][next_point_i][0], points[r][next_point_i][1], points[r-1][next_point_i][0], points[r-1][next_point_i][1]]);
          paths.push([points[r-1][next_point_i][0], points[r-1][next_point_i][1], points[r][i][0], points[r][i][1]]);
          polygons.push([
            points[r][i][0], 
            points[r][i][1], 
            points[r][next_point_i][0], 
            points[r][next_point_i][1], 
            points[r-1][next_point_i][0], 
            points[r-1][next_point_i][1],
            points[r][i][0], 
            points[r][i][1],
          ])
        } else {
          paths.push([points[r][i][0], points[r][i][1], points[r-1][next_point_i][0], points[r-1][next_point_i][1]]);
          paths.push([points[r-1][next_point_i][0], points[r-1][next_point_i][1], points[r][i][0], points[r][i][1]]);
          paths.push([points[r][i][0], points[r][i][1], points[r-1][i][0], points[r-1][i][1]]);
        }
        polygons.push([
                        points[r][i][0], 
                        points[r][i][1], 
                        points[r-1][next_point_i][0], 
                        points[r-1][next_point_i][1],
                        points[r-2][i][0], 
                        points[r-2][i][1], 
                        points[r-1][i][0], 
                        points[r-1][i][1],
                        points[r][i][0], 
                        points[r][i][1], 
                      ])
        
      }
    }

  }

  let svgString = '<svg xmlns="http://www.w3.org/2000/svg">';

  // paths.forEach(path => {
  //   svgString += `<line x1="${path[0]}" y1="${path[1]}" x2="${path[2]}" y2="${path[3]}" stroke="black" />`;
  //   svgString += `<line x1="${path[0]}" y1="${path[1]}" x2="${path[2]}" y2="${path[3]}" stroke="black" />`;
  // });

  const SCALE_FACTOR = 0.9;

  polygons.forEach(polygon => {
    const point_pairs : string[] = [];
    let x_points : number[] = [], 
        y_points : number[] = [],
        x_avg : number = 0,
        y_avg : number = 0;

    for (let i = 0; i + 1 < polygon.length; i+=2) {
      x_points.push(polygon[i]);
      y_points.push(polygon[i+1]);
      x_avg += polygon[i];
      y_avg += polygon[i+1];
      // point_pairs.push(`${polygon[i]} ${polygon[i+1]}`);
    }

    x_avg /= x_points.length;
    y_avg /= y_points.length;

    
    for (let i = 0; i + 1 < polygon.length; i += 2) {
      const x_diff = x_points[i/2] - x_avg;
      const y_diff = y_points[i/2] - y_avg;
      x_points[i/2] = x_avg + x_diff * SCALE_FACTOR;
      y_points[i/2] = y_avg + y_diff * SCALE_FACTOR;
      point_pairs.push(`${x_points[i/2]} ${y_points[i/2]}`);
    }
    point_pairs.push(`${x_points[0]} ${y_points[0]}`);

    // for (let i = 0; i + 1 < polygon.length; i+=2) {
    //   x_points.push(polygon[i]);
    //   y_points.push(polygon[i+1]);
    //   point_pairs.push(`${polygon[i]} ${polygon[i+1]}`);
    // }
    svgString += `<polygon points="${point_pairs.join(" ")}" fill="black" />`;
  });



  svgString += '</svg>';

  console.log("Created SVG")
  console.log(svgString);

};

export const QueryLakeLogoSvg = ({
  className = ""
}:{
  className?: string
}) => (
  <svg className={cn("subpixel-antialiased", className)} fill="currentColor" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 448 448">
    <g id="Logo">
      <path id="Arrow" d="m338.7936,445.85h107.0564v-107.0564c-23.7471,45.8002-61.2562,83.3093-107.0564,107.0564Z"/>
      <g id="Left">
        <g id="L1">
          <path d="m93.4572,146.066l208.478,208.4779c-23.4828,14.0502-50.3241,21.4542-77.9399,21.4542-40.5989,0-78.7678-15.8101-107.4755-44.5178-12.603-12.6031-22.7804-27.093-30.2494-43.0674-7.2188-15.4392-11.7562-31.91-13.4864-48.9549-3.3051-32.5608,4.0228-65.5264,20.6732-93.392m-.6869-26.3839c-6.5363,0-13.02,3.1788-16.9201,9.2431-43.8235,68.1412-35.9255,159.9011,23.699,219.5257,34.3649,34.3649,79.4059,51.5473,124.4461,51.5473,33.1064,0,66.2119-9.2827,95.0796-27.8483,10.7576-6.9185,12.4363-21.971,3.3923-31.0151L106.8652,125.5329c-3.9458-3.9457-9.0363-5.8508-14.095-5.8508h0Z"/>
        </g>
      </g>
      <g id="Right">
        <g id="L1-2" data-name="L1">
          <path d="m224.0172,48.002v24h.0009c40.5902,0,78.7544,15.8101,107.4621,44.5179,12.603,12.603,22.7804,27.0929,30.2494,43.0672,7.2188,15.4392,11.7562,31.91,13.4864,48.9549,3.3051,32.5608-4.0228,65.5265-20.6732,93.392L146.0648,93.4561c23.4801-14.0484,50.3216-21.4521,77.9417-21.4541l.0107-24m-.0125,0c-33.1017.0025-66.2161,9.2853-95.0796,27.8482-10.7576,6.9185-12.4363,21.971-3.3923,31.0151l215.6019,215.6019c3.9463,3.9463,9.0357,5.8508,14.095,5.8508,6.5356,0,13.0204-3.1794,16.9201-9.243,43.8235-68.1413,35.9255-159.9012-23.699-219.5257-34.3686-34.3686-79.4011-51.5507-124.4461-51.5473h0Z"/>
        </g>
      </g>
      <g id="Circle">
        <path d="m224,24c51.1844,0,102.3689,19.5262,141.4214,58.5787,78.1049,78.1049,78.1049,204.7379,0,282.8427-39.0524,39.0524-90.2369,58.5787-141.4214,58.5787s-102.3689-19.5262-141.4213-58.5787C4.4738,287.3166,4.4738,160.6836,82.5787,82.5787c39.0524-39.0525,90.2368-58.5787,141.4213-58.5787m0-24c-29.0819,0-57.4376,5.519-84.2792,16.4038-27.8022,11.2743-52.7373,27.829-74.1127,49.2043-21.3753,21.3753-37.93,46.3104-49.2043,74.1126C5.519,166.5625,0,194.9182,0,224s5.519,57.4376,16.4038,84.2792c11.2743,27.8023,27.829,52.7374,49.2043,74.1127,21.3753,21.3753,46.3104,37.93,74.1127,49.2043,26.8417,10.8848,55.1973,16.4038,84.2792,16.4038s57.4376-5.519,84.2792-16.4038c27.8022-11.2743,52.7373-27.829,74.1126-49.2043,21.3753-21.3752,37.93-46.3103,49.2043-74.1127,10.8848-26.8417,16.4038-55.1972,16.4038-84.2792s-5.519-57.4376-16.4038-84.2792c-11.2743-27.8022-27.829-52.7373-49.2043-74.1126s-46.3104-37.93-74.1126-49.2043C281.4376,5.519,253.0819,0,224,0h0Z"/>
      </g>
    </g>
  </svg>
);
