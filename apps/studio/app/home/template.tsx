'use client'

// ATTEMPT 1
import { motion } from "framer-motion";
import { usePathname } from "next/navigation";
import { useEffect } from "react";

export default function Template({ 
    children 
}: {
    children: React.ReactNode;
}) {
  const pathname = usePathname();
  // useEffect(() => {
  //   console.log("pathname changed to:", pathname);
  // }, [pathname]);
  
  return (
    <motion.div 
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ type: "spring", ease: "linear", bounce: 0, duration: 1, delay: 0}}
      layout
    >
      {children}
    </motion.div>
  );
}