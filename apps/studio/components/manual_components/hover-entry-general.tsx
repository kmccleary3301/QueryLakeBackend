// import AnimatedPressable from "./AnimatedPressable";
import AnimatedPressable from "./animated-pressable";
// import { Text, View } from "react-native";
import { useState, ReactNode } from "react";

type HoverEntryGeneralProps = {
  title: string,
  onPress?: () => void,
  textStyle?: React.CSSProperties,
  children: ReactNode,
}

export default function HoverEntryGeneral(props : HoverEntryGeneralProps) {
  const [hover, setHover] = useState(false);
  return (
    <AnimatedPressable 
      style={{
        borderRadius: 4,
        flexDirection: 'row',
        justifyContent: 'space-between',
        paddingRight: 4,
      }}
      onHover={setHover}
      onPress={() => {
        if (props.onPress) { props.onPress(); }
      }}
    >
      <div style={{
				paddingTop: 3,
				paddingBottom: 3,
				display: 'flex', 
				flex: 1, 
				paddingRight: 5
			}}>
        <span style={(props.textStyle)?props.textStyle:{
          // fontFamily: 'Inter-Regular',
          fontSize: 14,
          color: '#E8E3E3',
          paddingTop: 4,
					paddingBottom: 4,
          // paddingHorizontal: 4
        }}>
          {props.title}
        </span>
      </div>
      {(hover) && (
        <div style={{
					display: 'flex',
          alignSelf: 'center',
          justifyContent: 'center',
          flexDirection: 'row'
        }}>

          {props.children}
        </div>
      )}
    </AnimatedPressable>
  );
}