module.exports = {
  purge: ['./src/**/*.{js,jsx,ts,tsx}', './public/index.html'],
  darkMode: false, // or 'media' or 'class'
  theme: {
    extend: {
      width: {
        '300': '300px',  // defining custom width
      },
      height: {
        '300': '300px',  // defining custom height
      }
    },
  },
  variants: {
    extend: {},
  },
  plugins: [],
}
