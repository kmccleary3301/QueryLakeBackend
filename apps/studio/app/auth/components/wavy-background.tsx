"use client";
import { cn } from "@/lib/utils";
import React, { useEffect, useRef, useState } from "react";
import { createNoise3D } from "simplex-noise";

export const WavyBackground = ({
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
      animation_points: number = 10,
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
    "rgb(56, 189, 248)",
    "rgb(129, 140, 248)",
    "rgb(192, 132, 252)",
    "rgb(232, 121, 249)",
    "rgb(34, 211, 238)",
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
    for (i = 0; i < n; i++) {
      const lt = speedSet[i] * 0.001 * time_passed * 0.05;
      ctx.beginPath();
      ctx.lineWidth = waveWidth || 50;
      ctx.strokeStyle = waveColors[i % waveColors.length];
      
      
      // Define the control points for the Bezier curve
      const controlPoints = [];
      const canvasHeight = canvasRef.current?.height || 100;
      for (x = 0; x < w; x += interval) {
        const xNormalized = (x) / (w);
        const expControl = Math.exp(-Math.pow(50 * (xNormalized - 0.5) * wavePinchMiddle, 2));
        const pinchedAmplitude = waveAmplitude * canvasHeight * ((1 - wavePinchEnd) * Math.sin(Math.PI * xNormalized) * expControl + wavePinchEnd);

        var y = noise(x / 800, 2 * i, lt) * pinchedAmplitude;
        controlPoints.push({ x, y: y + h * 0.5 });
      }

      // Add a single point at the end of w
      const xEnd = w;
      const xNormalizedEnd = xEnd / w;
      const pinchedAmplitudeEnd = waveAmplitude * ((1 - wavePinchEnd) * Math.sin(Math.PI * xNormalizedEnd) + wavePinchEnd);
      const yEnd = noise(xEnd / 800, 2 * i, lt) * pinchedAmplitudeEnd;
      controlPoints.push({ x: xEnd, y: yEnd + h * 0.5 });
  
      // Draw the Bezier curve
      ctx.moveTo(controlPoints[0].x, controlPoints[0].y);
      let j;
      for (j = 0; j < controlPoints.length - 2; j++) {
        // ctx.strokeStyle = new_color_2;
        // console.log("New color:", new_color_2);
        const xc = (controlPoints[j].x + controlPoints[j + 1].x) / 2;
        const yc = (controlPoints[j].y + controlPoints[j + 1].y) / 2;
        ctx.quadraticCurveTo(controlPoints[j].x, controlPoints[j].y, xc, yc);
      }
      // curve through the last two points
      ctx.quadraticCurveTo(controlPoints[j].x, controlPoints[j].y, controlPoints[j+1].x,controlPoints[j+1].y);
      ctx.stroke();
      ctx.closePath();
    }
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
          ...(isSafari ? { filter: `blur(${blur}px)` } : {}),
        }}
      ></canvas>
      <div className={cn("relative z-10", className)} {...props}>
        {children}
      </div>
    </div>
  );
};
