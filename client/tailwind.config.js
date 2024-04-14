module.exports = {
  purge: ['./src/**/*.{js,jsx,ts,tsx}', './public/index.html'],
  darkMode: false, // or 'media' or 'class'
  safelist: [
    {
      pattern: /bg-(red|blue|green|yellow|purple|pink|gray|orange)-[50|100|200|300|400|500|600|700|800|900|950]/,
    },
    {
      pattern: /text-(white|black)/,
    }
  ],
  theme: {
    extend: {
      width: {
        '250': '250px',
        '520': '520px',
        '150': '150px'
      },
      height: {
        '200': '200px',
        '400': '400px',
      },
      keyframes: {
        underlineSlide: {
          '0%': { width: '0', transform: 'translateX(-50%)', left: '50%' },
          '100%': { width: '100%', transform: 'translateX(0)', left: '0%' },
        }
      },
      listStyleType: {
        none: 'none',
        disc: 'disc',  // default bullet style
        decimal: 'decimal',  // for numbers
        square: 'square',  // square bullets
      },
      animation: {
        slide: 'underlineSlide 0.5s forwards',
      },
      transitionProperty: {
        'height': 'height',
        'width': 'width',
      },
    },
  },
  variants: {
    extend: {},
  },
  plugins: [],
}
