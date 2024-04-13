import React from "react";
import styled, {keyframes} from "styled-components";  // Consolidate imports from styled-components

interface AnimatedTextGradientProps {
  text: string;  // Define the type of the 'text' prop
}

const AnimatedTextGradient: React.FC<AnimatedTextGradientProps> = ({text}) => {
  return <AnimatedGradientText className={"font-bold text-4xl"}>{text}</AnimatedGradientText>;
};


const gradient = keyframes`
    0% {
        background-position: 0 50%;
    }
    50% {
        background-position: 100% 50%;
    }
    100% {
        background-position: 0 50%;
    }
`;

const AnimatedGradientText = styled.h1`
    animation: ${gradient} 4s ease-in-out infinite;
    background: linear-gradient(to right bottom, #ae75ff, #8f40ff, #580cc4,  #38077d);
    background-size: 300%;
    background-clip: text;
    -webkit-text-fill-color: transparent;
`;

export default AnimatedTextGradient
