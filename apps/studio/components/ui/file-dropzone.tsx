import React, { useEffect, useRef, useState } from 'react';
import { Button } from './button';
import { cn } from '@/lib/utils';

export default function FileDropzone({ 
  onFile,
  multiple = false,
  style = {}
}:{
  onFile: (files: File[]) => void;
  multiple?: boolean;
  style?: React.CSSProperties;
}) {
  // const [isDragging, setIsDragging] = useState(false);
  // const [files, setFiles] = useState<File[]>([]);


  // useEffect(() => {
  //   if (files.length > 0) {
  //     onFileSelected(files[files.length - 1]);
  //     setFiles([...files.slice(0, files.length - 1)]);
  //   }
  // }, [files]);

  // const handleFileInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
  //   const file = event.target.files?.[0];
  //   if (file) {
  //     onFileSelected(file);
  //   }
  // };

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleButtonClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (files_in : FileList) => {
    if (files_in !== null && files_in.length > 0) {
      if (multiple) {
        onFile(Array.from(files_in));
      } else {
        onFile([Array.from(files_in)[0]]);
      }
    }
  };

  return (
    <div className={cn(
      "hover:bg-accent active:bg-accent/70 hover:text-accent-foreground hover:text-accent-foreground/",
      "rounded-lg border-[2px] border-dashed border-primary p-2 text-center cursor-pointer",
    )}
    onClick={handleButtonClick}
    onDrop={(e)=>{
      e.preventDefault();
      handleFileChange(e.dataTransfer.files);
    }}
    onDragOver={(e) => {
      e.preventDefault();
    }}
    style={style}
    >
      <div className='rounded-[inherit]'>
        <input
          type="file"
          multiple={multiple}
          ref={fileInputRef}
          style={{ display: 'none' }}
          onChange={(e) => {if (e.target.files) handleFileChange(e.target.files)}}
        />
        <p>Upload File</p>
      </div>
    </div>
  );
};