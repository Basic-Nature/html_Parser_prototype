module.exports = {
  root: true,
  env: {
    browser: true,
    node: true,
    es2021: true,
  },
  parserOptions: {
    ecmaVersion: 2021,
    sourceType: 'script'
  },
  extends: [ 'eslint:recommended' ],
  plugins: [ 'unused-imports' ],
  globals: {
    bootstrap: 'readonly',
    io: 'readonly',
    socket: 'readonly',
    getActiveSessionId: 'readonly',
    navigator: 'readonly'
  },
  rules: {
    // keep rules permissive initially; enforce stricter rules later
    'no-unused-vars': 'off',
    'unused-imports/no-unused-imports': 'error',
    'unused-imports/no-unused-vars': ['warn', {
      args: 'after-used',
      argsIgnorePattern: '^_',
      vars: 'all',
      varsIgnorePattern: '^_'
    }],
    'no-console': 'off',
    // allow empty catch blocks (some code uses silent catch intentionally)
    'no-empty': ['error', { 'allowEmptyCatch': true }],
    // some regexes contain control characters from data; relax for now
    'no-control-regex': 'off'
  }
};

