"use client";
import { useEffect } from "react";
import { motion, stagger, useAnimate } from "framer-motion";

export const TextGenerateEffect = ({
  words,
  className,
  staggerDelay = 0.4,
  initialDelay = 0,
  duration = 1,
  byLetter = false,
  spring = false,
}: {
  words: string;
  className?: string
  staggerDelay?: number;
  initialDelay?: number;
  duration?: number;
  byLetter?: boolean;
  spring?: boolean;
}) => {
  const [scope, animate] = useAnimate();
  let wordsArray = byLetter ? words.split("") : words.split(" ");
  useEffect(() => {
    if (scope.current) {
      const timeoutId = setTimeout(() => {

        animate(
          "span",
          {
            opacity: 1,
          },
          {
            type: spring ? "spring" : "tween",
            duration: duration,
            delay: stagger(staggerDelay),
          }
        );
      }, 1000*initialDelay);

      return () => {
        clearTimeout(timeoutId);
      };

    }
  }, [animate, duration, initialDelay, spring, staggerDelay, words, byLetter, scope]);

  const renderWords = () => {
    return (
      <motion.div ref={scope}>
        {wordsArray.map((word, idx) => {
          return (
            <motion.span
              key={word + idx}
              className="opacity-0"
            >
              {word}{byLetter?"":" "}
            </motion.span>
          );
        })}
      </motion.div>
    );
  };

  return (
    <div className={className}>
      {renderWords()}
    </div>
  );
};
