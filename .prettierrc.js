module.exports = {
  // General
  semi: true,
  singleQuote: true,
  printWidth: 88,
  useTabs: false,
  trailingComma: "es5",
  bracketSpacing: true,

  // HTML, CSS e SCSS
  htmlWhitespaceSensitivity: 'css',
  cssEnable: true,
  scssEnable: true,

  // Markdown
  proseWrap: 'always',

  // YAML
  tabWidth: 2,

  // JSON
  overrides: [
    {
      files: "*.json",
      options: {
        tabWidth: 2,
        singleQuote: false,
      },
    },
  ],
};
