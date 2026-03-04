import { ReactNode } from 'react';
import { Button } from '../ui/button';

type AnimatedPressableProps = {
	style?: React.CSSProperties,
	onPress?: () => void,
  children: ReactNode,
	hoverColor?: string,
  pressColor?: string,
  onHover?: (hovering : boolean) => void,
  invert?: boolean,
  disableOpacity?: boolean
}

export default function AnimatedPressable(props: AnimatedPressableProps) {
	// const [hover, setHover] = useState(false);
  // const [pressed, setPressed] = useState(false);
  // const anim = useMemo(() => new Animated.Value(0), [color]);
	
  const handleDragOver = (event: React.MouseEvent<HTMLDivElement, MouseEvent>) => {
    // setHover(true);
    event.preventDefault();
    if (props.onHover) { props.onHover(true); }
  };

  const handleDragEnd = (event: React.MouseEvent<HTMLDivElement, MouseEvent>) => {
    // setHover(false);
    event.preventDefault();
    if (props.onHover) { props.onHover(false); }
  };

	return (
		<div
			// className={(props.style?.backgroundColor)?"bg-["+props.style.backgroundColor+"]/20 hover:bg-["+props.style.backgroundColor+"]/0":"bg-primary hover:bg-primary/90"}
      onMouseEnter={handleDragOver}
      onMouseLeave={handleDragEnd}
    >
        <Button variant={"ghost"}
          style={(props.style)?props.style:{}}
          onClick={() => {
            if (props.onPress) {
              props.onPress();
            }
          }}
        >
          {props.children}
        </Button>
    </div>
	);
}