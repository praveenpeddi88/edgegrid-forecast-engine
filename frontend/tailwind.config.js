/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // EdgeGrid brand (spec §Part IV)
        teal: {
          DEFAULT: "#0E7C7B",
          50:  "#E6F4F4",
          500: "#0E7C7B",
          600: "#0A6867",
          700: "#085352",
        },
      },
      fontFamily: {
        sans: ['Poppins', 'ui-sans-serif', 'system-ui', 'sans-serif'],
      },
      transitionDuration: {
        fast: '150ms',
      },
    },
  },
  plugins: [],
};
