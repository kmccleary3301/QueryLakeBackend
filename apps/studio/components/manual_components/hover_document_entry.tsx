import { useState } from "react";
import * as Icon from 'react-feather';

// import { Feather } from "@expo/vector-icons";
import { Button } from "@/components/ui/button";

type HoverDocumentEntryProps = {
  deleteIndex: () => void,
  title: string,
  onPress?: () => void,
  iconColor?: string,
  textStyle?: React.CSSProperties,
  style?: React.CSSProperties,
  disableOpacity?: boolean
}

export default function HoverDocumentEntry(props : HoverDocumentEntryProps) {
  const [hover, setHover] = useState(false);
  
  return (
    <div 
      // disableOpacity={(props.disableOpacity)?props.disableOpacity:false}
      style={{...{
        borderRadius: 4,
        paddingVertical: 1,
				display: "flex",
        flexDirection: 'row',
        justifyContent: 'space-around',
        paddingRight: 4,
        maxWidth: "100%",
        height: 28,
        // borderWidth: 2,
        // borderColor: "#FF0000",
      }, ...(props.style)?props.style:{}}}
      onMouseEnter={() => {setHover(true);}}
			onMouseLeave={() => {setHover(false);}}
      onClick={() => {
        if (props.onPress) { props.onPress(); }
      }}
    >
      <div style={{...{
        flex: 1, 
        maxWidth: '100%',
        height: '100%',
        alignContent: 'center',
      }}}>
        <div style={{
					display: "flex",
          justifyContent: 'center',
          alignSelf: 'center',
          // borderWidth: 2,
          // borderColor: "#FF0000",
          height: 28,
          width: '100%'
        }}>
          <Button variant="link" style={{...{
            // fontFamily: 'Inter-Regular',
            fontSize: 14,
            color: '#E8E3E3',
            // paddingVertical: 2,
            textAlign: 'left',
            maxWidth: "100%",
						height: "100%",
						// width: "100%",
						flex: 1,
						display: "flex",
						justifyContent: "flex-start",	
            // paddingHorizontal: 4
          }, ...(props.textStyle)?props.textStyle:{}}}>
            {props.title}
          </Button>
        </div>
      </div>

      {(hover) && (
        <div style={{
					display: "flex",
          justifyContent: 'center',
          alignSelf: 'center',
          // borderWidth: 2,
          // borderColor: "#FF0000",
          // height: 30,
          paddingLeft: 10
        }}>
          <Button variant="ghost" className="hover:bg-accent/0 hover:text-accent-none" onClick={props.deleteIndex} style={{
						display: "flex",
						flexDirection: 'column', 
						alignContent: 'center',
						// paddingBottom: 0,
						// paddingTop: 0,
						padding: 0,
						// backgroundColor: "#23232D",
					}}>
            <Icon.Trash size={14} color={(props.iconColor)?props.iconColor:'#E8E3E3'}/>
          </Button>
        </div>
      )}
    </div>
  );
}