import axios from 'axios';
import craftUrl from './craftUrl';

export default async function uploadFiles({
  files,
  url,
  parameters,
  on_upload_progress = () => {},
  on_response = () => {},
}:{
  files: File[], 
  url: string, 
  parameters: object,
  on_upload_progress?: (progress: number, index: number) => void,
  on_response?: (response: object, index: number) => void,
}) : Promise<Object[]> {
  let responses : Object[] = [];

  const upload_endpoint = craftUrl(url, parameters);

  for (let i = 0; i < files.length; i++) {
    const file = files[i];
    const formData = new FormData();
    formData.append("file", file);
    
    try {
      const response = await axios.post(upload_endpoint, formData, {
        onUploadProgress: (progressEvent) => {
          const progress = Math.round((progressEvent.loaded / (progressEvent.total || 1)) * 100);
          on_upload_progress(progress, i);
        },
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });

      const response_data = response.data;
      on_response(response_data, i);
      responses.push(response_data);
    } catch (error) {
      console.error(`Failed to upload file ${file.name}:`, error);
      responses.push({error: error});
    }
  }

  return responses;
}