import 'package:flutter_test/flutter_test.dart';
import 'package:voice_gender_app/main.dart';

void main() {
  testWidgets('Voice Gender App loads', (WidgetTester tester) async {
    await tester.pumpWidget(const VoiceGenderApp());

    expect(find.text('Voice Gender Classification'), findsWidgets);
    expect(find.text('Upload Audio'), findsOneWidget);
  });
}