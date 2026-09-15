// Server-side math rendering: the browser needs only local CSS and fonts.
const fs = require('fs');
const katex = require(process.argv[2]);
const fragments = JSON.parse(fs.readFileSync(0, 'utf8'));
const rendered = fragments.map(({text, display}) => katex.renderToString(text, {
  displayMode: display, throwOnError: true, trust: false, output: 'htmlAndMathml'
}));
process.stdout.write(JSON.stringify(rendered));
